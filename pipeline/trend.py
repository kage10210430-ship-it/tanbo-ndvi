"""生育傾向マップ用のデータを作る（稲の年だけを重ねて、田の中で毎年どこが強い・弱いかを見る）。

福井は2年3作（稲→麦→大豆）が多く、同じ田でも年によって作物が違う。作物をまぜて平均すると、
湿った所（稲はよく育ち、麦・大豆は弱る）の傾向が打ち消し合ったり逆になったりするので、
年ごとに作物の手がかりを取っておき、表示側で「稲の年」だけを重ねる。

各セル・各年（config の trend_from 年から、8月末を過ぎた年まで）について:
  1) 区画ごとの作物の手がかり（区画の内側 inner.geojson の中央値）
       A = 4月の NDVI（麦があると高い）
       J = 7/10〜7/31 の NDVI（稲は高い。麦のあとの大豆はまだ低い）
       R = 5/1〜6/10 のレーダー VH（dB）（田に水が張ってあると低い。雲に関係なく見える）
  2) 6月ごろ・8月ごろ（config の trend_windows）の晴れた日ごとに、区画の中の各画素の NDVI と
     その日のその区画の中央値との差を出し（区画の8割以上が晴れた日だけ）、画素ごとに日をまたいだ中央値をとる
     → その年・その時期の「田の中の差」
稲の年かどうかは表示側（index.html）で決める（しきい値を変えても計算し直さなくてよいように）。
  いまの基準（index.html の cropOf）: 麦 = A≧0.6。稲 = J≧0.5 かつ（R≦−21dB、または R≦−18dB で6月ごろの田の中央値 NDVI<0.6）

  python pipeline/trend.py --minutes 300 --workers 4
1セル・1年ずつ保存するので、時間切れで止まっても次回に続きから作る。できあがった年は作り直さない。
--backend fake を付けると GEE を使わずに作り物のデータで作る（動作確認用）。

cells/<id>/trend.png: 年・時期ごとの「田の中の差」を縦に積んだグレースケール
                      （0=なし, 1〜255=差 -0.3〜+0.3）。格子は px_<年>.png と同じ（EPSG:3857）
cells/<id>/trend.json:
  {"v":1, "mask":"<雲判定の方式>", "w","h","x0","y1","px", "win":{"e":[始め,終わり],"l":[...]},
   "years":[済んだ年], "layers":["2019e","2019l",...]（PNG に入っている順）,
   "p":{"<pid>":{"2019":[A,J,R,ne,me,nl,ml], ...}}}
     A,J = NDVI×1000, R = dB×10（なければ null）, ne/nl = 6月ごろ・8月ごろに使った日数,
     me/ml = その時期のその区画の中央値の平均（NDVI×1000。使った日がなければ null）
区画の縁（約8m）の画素は畦・道路が混ざって毎年低く出るので使わない（0 のまま。表示では内側の値でうめる）。
"""
import os, math, time, hashlib, datetime as dt
from concurrent.futures import ThreadPoolExecutor, as_completed
from common import write_json, read_json, with_retry
from pixels import grid_for, read_layers, write_png

VERSION = 1
DEV = 0.3                 # 保存する差の範囲（±NDVI）
MIN_PARCEL_CLEAR = 0.8    # その日に区画の画素のこれ以上が晴れていれば使う
MIN_CELL_CLEAR = 0.2      # セルの範囲でこれ未満しか晴れていない日は画素を取らない
EDGE_PX = (0.8, 0.4)      # 区画の縁からこの画素数（≒8m）以内の画素は使わない（畦・道路が混ざって毎年低く出るため。表示では内側の値でうめる）。
                          # 小さい田で MIN_INNER_PX 画素も残らないときは、縁を4mだけ除く
MIN_INNER_PX = 8
DEFAULT_WINDOWS = {"e": ["06-01", "07-05"], "l": ["07-20", "08-20"]}
HIST_WINDOW = ("05-10", "10-01")    # 過去の年の推移（グラフ用）を取る期間（終わりの日は含まない）


def trend_years(cfg, today):
    """作る年の一覧（8月末を過ぎた年まで）"""
    last = today.year if today >= dt.date(today.year, 9, 1) else today.year - 1
    return list(range(int(cfg.get("trend_from", 2022)), last + 1))


def windows(cfg):
    return cfg.get("trend_windows") or DEFAULT_WINDOWS


class GEETrend:
    """GEE から作物の手がかりと晴れた日の画素を取る（backend は update_ndvi.GEEBackend, px は pixels.GEEPixels）"""
    def __init__(self, backend, px):
        self.b = backend; self.px = px; self.ee = backend.ee

    def _median(self, col, band):
        """画素ごとの中央値。画像が1枚もなくてもエラーにしない（全部マスクされた画像になる）"""
        ee = self.ee
        base = ee.Image.constant(0).toFloat().rename(band).updateMask(0)
        return ee.ImageCollection([base]).merge(col.map(lambda i: i.select([0]).rename(band).toFloat())).median().rename(band)

    def features(self, bbox, inner, year):
        """{pid: [A, J, R]}（値がなければ None）"""
        ee = self.ee; y = year
        img = ee.Image.cat([
            self._median(self.b._col(bbox, f"{y}-04-01", f"{y}-05-01", "s2"), "A"),
            self._median(self.b._col(bbox, f"{y}-07-10", f"{y}-08-01", "s2"), "J"),
            self._median(self.b._raw(bbox, f"{y}-05-01", f"{y}-06-11", "s1").select("VH"), "R")])
        # 中央値にするのは、畦・道路が混ざった縁の画素（レーダーでは特に明るい）に引っぱられないように
        table = img.reduceRegions(collection=ee.FeatureCollection(inner), reducer=ee.Reducer.median(), scale=10, tileScale=2)
        res = with_retry(lambda: table.select(["pid", "A", "J", "R"], None, False).getInfo())
        return {f["properties"]["pid"]: [f["properties"].get(k) for k in ("A", "J", "R")] for f in res["features"]}

    def clear_dates(self, bbox, start, end, min_frac):
        """期間内で、セルの範囲の min_frac 以上が晴れている日（Cloud Score+ で判定）"""
        ee = self.ee
        region = ee.Geometry.Rectangle(bbox); thr = self.b.cfg.get("cloud_score_min", 0.6)
        fc = self.b._csp(bbox, start, end).map(lambda i: ee.Feature(None, {
            "t": i.get("system:time_start"),
            "c": i.select("cs_cdf").gte(thr).unmask(0).reduceRegion(ee.Reducer.mean(), region, 60, bestEffort=True).get("cs_cdf")}))
        info = with_retry(lambda: fc.getInfo())
        frac = {}
        for f in info["features"]:
            p = f["properties"]
            d = dt.datetime.fromtimestamp(p["t"] / 1000, dt.timezone.utc).strftime("%Y-%m-%d")
            frac[d] = min(1.0, frac.get(d, 0) + (p.get("c") or 0))    # 同じ日の隣のタイルは足す
        return sorted(d for d, c in frac.items() if c >= min_frac)

    def ndvi(self, bbox, dates, g):
        return self.px.ndvi(bbox, dates, g)


class FakeTrend:
    """動作確認用（GEE を使わない）。3枚に1枚は2年3作（1年おきに麦→大豆）、残りは毎年稲。
    田の中の模様: セルの中の円（深い所）は6月は弱く8月は強い。東ほど少し強い"""
    def features(self, bbox, inner, year):
        out = {}
        for f in inner["features"]:
            pid = f["properties"]["pid"]; h = int(hashlib.md5(pid.encode()).hexdigest()[:8], 16)
            rice = h % 3 != 0 or (year + h) % 2 == 0
            out[pid] = [0.22, 0.78, -25.5] if rice else [0.76, 0.31, -18.2]
        return out

    def clear_dates(self, bbox, start, end, min_frac):
        d0, d1 = dt.date.fromisoformat(start), dt.date.fromisoformat(end)
        return [(d0 + dt.timedelta(days=k)).isoformat() for k in range(2, (d1 - d0).days, 6)]

    def ndvi(self, bbox, dates, g):
        import numpy as np
        yy, xx = np.mgrid[0:g["h"], 0:g["w"]]
        deep = (xx - g["w"] * 0.4) ** 2 + (yy - g["h"] * 0.5) ** 2 < (min(g["w"], g["h"]) * 0.25) ** 2
        out = {}
        for d in dates:
            early = d[5:] < "07-10"
            rng = np.random.default_rng(int(d.replace("-", "")) + int(abs(bbox[0]) * 1000))
            v = (0.45 if early else 0.8) + 0.04 * (xx / g["w"] - 0.5) + np.where(deep, -0.08 if early else 0.05, 0) + rng.normal(0, 0.02, xx.shape)
            a = np.clip(np.round((np.clip(v, -0.2, 1.0) + 0.2) / 1.2 * 254) + 1, 1, 255).astype(np.uint8)
            if rng.random() < 0.3:                   # ときどき雲（南の3分の1）
                a[int(g["h"] * 2 / 3):] = 0
            out[d] = a
        return out


def labels(parcels, g, edges=EDGE_PX):
    """区画の番号を塗った格子（0=区画の外・縁, k=parcels の k 番目（1から））。
    画素の中心が、区画を edges[0] 画素だけ内側に縮めた形の中にあるかで決める（MIN_INNER_PX 画素も残らなければ次の幅で）。
    edges=(0,) なら縮めない（表示側 drawPx と同じ判定）"""
    import numpy as np, shapely
    from shapely.geometry import shape
    lab = np.zeros((g["h"], g["w"]), np.int32)
    def to_grid(xy):
        x = 6378137.0 * np.radians(xy[:, 0]); y = 6378137.0 * np.log(np.tan(np.pi / 4 + np.radians(xy[:, 1]) / 2))
        return np.column_stack([(x - g["x0"]) / g["px"], (g["y1"] - y) / g["px"]])
    for k, f in enumerate(parcels["features"], 1):
        try:                                         # 形がこわれた区画があっても、そのセル全体を止めない
            poly = shapely.transform(shape(f["geometry"]), to_grid)
            for i, edge in enumerate(edges):
                gg = poly.buffer(-edge) if edge else poly.buffer(0)
                if gg.is_empty:
                    continue
                x0, y0, x1, y1 = gg.bounds
                c0, c1 = max(0, math.floor(x0)), min(g["w"], math.ceil(x1))
                r0, r1 = max(0, math.floor(y0)), min(g["h"], math.ceil(y1))
                if c1 <= c0 or r1 <= r0:
                    continue
                xs, ys = np.meshgrid(np.arange(c0, c1) + 0.5, np.arange(r0, r1) + 0.5)
                sub = lab[r0:r1, c0:c1]
                hit = shapely.contains_xy(gg, xs, ys) & (sub == 0)
                if hit.sum() >= MIN_INNER_PX or i == len(edges) - 1:
                    sub[hit] = k
                    break
        except Exception:
            continue
    return lab


def label_median(ls, vs, n):
    """区画番号 ls ごとの vs の中央値（長さ n+1、値がない区画は NaN）"""
    import numpy as np
    order = np.lexsort((vs, ls)); ls = ls[order]; vs = vs[order]
    cnt = np.bincount(ls, minlength=n + 1); start = np.concatenate([[0], np.cumsum(cnt)[:-1]])
    med = np.full(n + 1, np.nan)
    has = cnt > 0
    med[has] = (vs[start[has] + (cnt[has] - 1) // 2] + vs[start[has] + cnt[has] // 2]) / 2
    return med


def window_dev(arrays, lab, npx):
    """1つの時期の日ごとの画素（{日: uint8 配列, 0=雲}）から、画素ごとに
    「その日のその区画の中央値との差」の、日をまたいだ中央値を出す。
    戻り値: (差 float32 の配列（なしは NaN）または None, 区画ごとの使った日数, 区画ごとの中央値の合計)"""
    import numpy as np
    n = len(npx) - 1
    used = np.zeros(n + 1, np.int32); medsum = np.zeros(n + 1)
    need = np.maximum(3, np.ceil(MIN_PARCEL_CLEAR * npx))
    inp = lab > 0; devs = []
    for d in sorted(arrays):
        a = arrays[d]
        ok = inp & (a > 0)
        good = (np.bincount(lab[ok], minlength=n + 1) >= need) & (npx > 0)
        good[0] = False
        if not good.any():
            continue
        sel = ok & good[lab]
        ls = lab[sel]; vs = (a[sel].astype(np.float32) - 1) / 254 * 1.2 - 0.2
        med = label_median(ls, vs, n)
        dev = np.full(lab.shape, np.nan, np.float32); dev[sel] = vs - med[ls]
        devs.append(dev); used[good] += 1; medsum[good] += med[good]
    if not devs:
        return None, used, medsum
    st = np.stack(devs); none = np.isnan(st).all(axis=0)
    st[:, none] = 0                                  # どの日も使えなかった画素は、中央値をとってから NaN に戻す（警告を出さない）
    out = np.nanmedian(st, axis=0); out[none] = np.nan
    return out, used, medsum


def encode(dev):
    """差（NaN=なし）→ uint8（0=なし, 1〜255=-DEV〜+DEV）"""
    import numpy as np
    out = np.zeros(dev.shape, np.uint8); ok = ~np.isnan(dev)
    out[ok] = np.clip(np.round((np.clip(dev[ok], -DEV, DEV) + DEV) / (2 * DEV) * 254) + 1, 1, 255).astype(np.uint8)
    return out


def meta_ok(meta, cfg, mask):
    return meta.get("v") == VERSION and meta.get("mask") == mask and meta.get("win") == windows(cfg)


def update_cell(src, cfg, data_dir, cell, today, mask, log, deadline=None):
    """1セルの生育傾向データを、まだ作っていない年だけ足す（1年ごとに保存）。作った年数を返す。"""
    import numpy as np
    cdir = os.path.join(data_dir, "cells", cell["id"])
    parcels = read_json(os.path.join(cdir, "parcels.geojson"))
    inner = read_json(os.path.join(cdir, "inner.geojson"))
    if not parcels or not parcels.get("features") or not inner or not inner.get("features"):
        return 0
    g = grid_for(cell["bbox"]); win = windows(cfg)
    meta_p, png_p = os.path.join(cdir, "trend.json"), os.path.join(cdir, "trend.png")
    meta = read_json(meta_p) or {}
    if not meta_ok(meta, cfg, mask) or any(meta.get(k) != g[k] for k in g):
        meta = {}                                    # 方式・時期・格子が変わったら作り直し
    todo = [y for y in trend_years(cfg, today) if y not in set(meta.get("years", []))]
    if not todo:
        return 0
    layers = read_layers(png_p, {"dates": meta.get("layers", []), "h": g["h"]}) if meta else {}
    lab = labels(parcels, g)
    pids = [f["properties"]["pid"] for f in parcels["features"]]
    npx = np.bincount(lab.ravel(), minlength=len(pids) + 1)
    p = meta.get("p", {}); years = set(meta.get("years", [])); made = []
    for y in todo:
        if deadline and time.time() > deadline:
            break
        feats = src.features(cell["bbox"], inner, y)
        first = min(win[w][0] for w in win); last = max(win[w][1] for w in win)
        clear = src.clear_dates(cell["bbox"], f"{y}-{first}", (dt.date.fromisoformat(f"{y}-{last}") + dt.timedelta(days=1)).isoformat(), MIN_CELL_CLEAR)
        rec = {}
        for w in ("e", "l"):
            dates = [d for d in clear if f"{y}-{win[w][0]}" <= d <= f"{y}-{win[w][1]}"]
            dev, used, medsum = window_dev(src.ndvi(cell["bbox"], dates, g) if dates else {}, lab, npx)
            if dev is not None:
                layers[f"{y}{w}"] = encode(dev)
            else:
                layers.pop(f"{y}{w}", None)
            rec[w] = (used, medsum)
        for k, pid in enumerate(pids, 1):
            A, J, R = (feats.get(pid) or [None] * 3)
            row = [None if A is None else round(A * 1000), None if J is None else round(J * 1000), None if R is None else round(R * 10)]
            for w in ("e", "l"):
                n = int(rec[w][0][k]); row += [n, round(rec[w][1][k] / n * 1000) if n else None]
            if any(v is not None for v in row[:3]) or row[3] or row[5]:
                p.setdefault(pid, {})[str(y)] = row
        years.add(y); made.append(y)
        keys = sorted(layers)
        if keys:
            write_png(png_p, [layers[k] for k in keys])
        elif os.path.exists(png_p):
            os.remove(png_p)
        write_json(meta_p, {"v": VERSION, "mask": mask, **g, "win": win, "years": sorted(years), "layers": keys, "p": p})
    if made:
        log(f"{cell['id']}: 生育傾向 {made[0]}" + (f"〜{made[-1]}" if len(made) > 1 else "") + f"年（{len(made)}年分, 区画{len(pids)}）")
    return len(made)


def hist_years(cfg, today, nd):
    """hist.json に入れる年: 生育傾向の年のうち、5/10〜9/30 が ndvi.json に入っていない年"""
    first = (nd or {}).get("dates", [""])[0] if (nd or {}).get("dates") else "9999"
    return [y for y in trend_years(cfg, today) if f"{y}-{HIST_WINDOW[0]}" < first]


def update_hist(backend, cfg, data_dir, cell, today, mask, log, deadline=None):
    """過去の年（ndvi.json より前）の 5/10〜9/30 の区画平均 NDVI を hist.json に足す（生育傾向のグラフ用）。
    形式は ndvi.json と同じ v2 に "years"（済んだ年）を足したもの。作った年数を返す"""
    from update_ndvi import update_source, load_full
    cdir = os.path.join(data_dir, "cells", cell["id"])
    inner = read_json(os.path.join(cdir, "inner.geojson"))
    if not inner or not inner.get("features"):
        return 0
    want = hist_years(cfg, today, read_json(os.path.join(cdir, "ndvi.json")))
    h = read_json(os.path.join(cdir, "hist.json")) or {}
    if h.get("v") != 2 or h.get("mask") != mask:
        h = {"v": 2, "mask": mask, "dates": [], "p": {}, "years": []}
    todo = [y for y in want if y not in h["years"]]
    full = load_full(cdir); made = []
    for y in todo:
        if deadline and time.time() > deadline:
            break
        update_source(backend, cfg, cdir, "hist.json", h, cell, inner, full, "s2", f"{y}-{HIST_WINDOW[0]}", f"{y}-{HIST_WINDOW[1]}")
        h["years"] = sorted(set(h["years"]) | {y}); made.append(y)
        write_json(os.path.join(cdir, "hist.json"), h)
    if made:
        log(f"{cell['id']}: 過去の推移 {made[0]}" + (f"〜{made[-1]}" if len(made) > 1 else "") + "年")
    return len(made)


def order_cells(cfg, cells):
    """config の trend_center（経度, 緯度）に近いセルから"""
    c = cfg.get("trend_center")
    if not c:
        return list(cells)
    k = math.cos(math.radians(c[1]))
    return sorted(cells, key=lambda x: math.hypot(((x["bbox"][0] + x["bbox"][2]) / 2 - c[0]) * k, (x["bbox"][1] + x["bbox"][3]) / 2 - c[1]))


def run_all(src, cfg, data_dir, cells, today, mask, log, limit, workers, task=None, need=None, name="生育傾向"):
    """まだ作っていない年があるセルを、時間（limit 秒）の許すかぎり作る。作った年数の合計を返す。
    task(cell, deadline) と need(cell) を渡すと、ほかのデータ（過去の年の推移 hist.json）にも使える"""
    t1 = time.time(); deadline = t1 + limit
    want = set(trend_years(cfg, today))
    if need is None:
        def need(c):
            m = read_json(os.path.join(data_dir, "cells", c["id"], "trend.json")) or {}
            return not (meta_ok(m, cfg, mask) and want <= set(m.get("years", [])))
    if task is None:
        def task(c, dl):
            return update_cell(src, cfg, data_dir, c, today, mask, log, dl)
    todo = [c for c in order_cells(cfg, cells) if need(c)]
    if not todo:
        log(f"{name}: すべてのセルが最新です"); return 0
    log(f"{name}: 作るセル {len(todo)}（{min(want)}〜{max(want)}年）")
    made = done = fails = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {}; pending = list(todo)
        while pending or futs:
            while pending and len(futs) < workers and time.time() < deadline:
                c = pending.pop(0)
                futs[ex.submit(task, c, deadline)] = c["id"]
            if not futs:
                break
            for f in as_completed(list(futs)):
                cid = futs.pop(f)
                try:
                    made += f.result(); done += 1
                except Exception as e:         # 1セルの失敗で全体を止めない（次回やり直す）
                    log(f"{cid}: {name} エラー {e}"); fails += 1
                break
            if not done and fails >= 2 * workers and pending:   # 最初から続けて失敗するときは、同じ失敗をくり返さないよう今回はやめる
                log(f"{name}: 最初の {fails} セルが続けて失敗したので、今回は中止します"); pending = []
            if time.time() >= deadline and pending:
                log(f"{name} 時間切れ。残り {len(pending)} セルは次回に続けます。"); pending = []
    log(f"{name} 完了: {done}セル, {made}年分（{(time.time() - t1) / 60:.1f}分）")
    return made


def main():
    import argparse, sys
    from common import load_config
    from update_ndvi import mask_id
    ap = argparse.ArgumentParser(description="生育傾向マップのデータを作る")
    ap.add_argument("--minutes", type=float, default=300, help="使う時間の上限（分）")
    ap.add_argument("--deadline", type=float, help="この時刻（UNIX 秒）までに終える（ワークフローの時間切れ対策）")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--backend", choices=["gee", "fake"], default="gee")
    ap.add_argument("--cells", help="カンマ区切りでセルIDを限定（テスト用）")
    args = ap.parse_args()
    def log(m): print(time.strftime("%H:%M:%S"), m, flush=True)
    cfg = load_config()
    data_dir = os.path.join(cfg["site_dir"], "data")
    index = read_json(os.path.join(data_dir, "index.json"))
    if not index:
        sys.exit("index.json がありません。先に build_cells.py を実行してください。")
    made = 0
    if not cfg.get("trend_maps"):
        log("生育傾向マップは作らない設定です（config の trend_maps）")
    else:
        if args.backend == "gee":
            from update_ndvi import GEEBackend
            from pixels import GEEPixels
            b = GEEBackend(cfg); src = GEETrend(b, GEEPixels(b))
        else:
            src = FakeTrend()
        limit = args.minutes * 60
        if args.deadline:
            limit = min(limit, args.deadline - time.time())
        want = set(args.cells.split(",")) if args.cells else None
        cells = [c for c in index["cells"] if want is None or c["id"] in want]
        if limit > 60:
            today = dt.datetime.now(dt.timezone.utc).date(); t2 = time.time()
            try:
                made = run_all(src, cfg, data_dir, cells, today, mask_id(cfg), log, limit, args.workers)
            except Exception as e:               # 途中まで作った分は公開する（-1 = 途中で止まった）
                log(f"生育傾向 エラー {e}"); made = -1
            # 過去の年の推移（生育傾向のグラフ用）。GEE のときだけ
            if args.backend == "gee" and limit - (time.time() - t2) > 60:
                mid = mask_id(cfg)
                def hneed(c):
                    cdir = os.path.join(data_dir, "cells", c["id"])
                    h = read_json(os.path.join(cdir, "hist.json")) or {}
                    return not (h.get("v") == 2 and h.get("mask") == mid and
                                set(hist_years(cfg, today, read_json(os.path.join(cdir, "ndvi.json")))) <= set(h.get("years", [])))
                try:
                    hm = run_all(src, cfg, data_dir, cells, today, mid, log, limit - (time.time() - t2), args.workers,
                                 task=lambda c, dl: update_hist(b, cfg, data_dir, c, today, mid, log, dl), need=hneed, name="過去の推移")
                    made = made + hm if made >= 0 else made
                except Exception as e:
                    log(f"過去の推移 エラー {e}"); made = made or -1
            # 獣害の起きやすさの手がかり（森との接し方・周りの田の割合）。区画の形が変わらなければ1回だけ
            if limit - (time.time() - t2) > 60:
                import wild
                wsrc = wild.GEEWild(b, src.px) if args.backend == "gee" else wild.FakeWild()
                try:
                    wm = run_all(src, cfg, data_dir, cells, today, mask_id(cfg), log, limit - (time.time() - t2), args.workers,
                                 task=lambda c, dl: wild.update_cell(wsrc, cfg, data_dir, index, c, log, dl),
                                 need=lambda c: wild.need(data_dir, c), name="獣害の手がかり")
                    made = made + wm if made >= 0 else made
                except Exception as e:
                    log(f"獣害の手がかり エラー {e}"); made = made or -1
            # 田の形（変形田の手がかり）。区画の形だけから作るので GEE は使わない。形が変わらなければ1回だけ
            import shape
            try:
                sm = run_all(src, cfg, data_dir, cells, today, mask_id(cfg), log, max(60, limit - (time.time() - t2)), args.workers,
                             task=lambda c, dl: shape.update_cell(cfg, data_dir, c), need=lambda c: shape.need(data_dir, c), name="田の形")
                made = made + sm if made >= 0 else made
            except Exception as e:
                log(f"田の形 エラー {e}"); made = made or -1
        else:
            log("生育傾向: 残り時間がないので今回は作りません")
    if os.environ.get("GITHUB_OUTPUT"):            # ワークフローで、作った年があるときだけ公開し直すため
        with open(os.environ["GITHUB_OUTPUT"], "a") as f:
            f.write(f"made={made}\n")


if __name__ == "__main__":
    main()
