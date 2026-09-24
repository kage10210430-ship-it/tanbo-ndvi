"""各セルの区画について、まだ計算していない観測日の NDVI（区画平均）を
Google Earth Engine で計算し、cells/<id>/ndvi.json に追記する。
Sentinel-2（10m）に加えて、Landsat 8/9（HLS L30, 30m）も使う（config の landsat）。

  python pipeline/update_ndvi.py --max-minutes 300 --workers 4

ndvi.json の形式（v2）:
  {"v": 2, "mask": "<雲判定の方式>",
   "dates": ["2025-09-25", ...],                     # 計算済みの観測日（昇順）
   "p": {"<pid>": [i, ndvi1000, pct, i, ndvi1000, pct, ...]}}
    i = dates の添字, ndvi1000 = NDVI×1000 の整数, pct = 有効画素率(%)
  Landsat は "lmask", "ldates", "lp" に同じ形で入れる（Sentinel-2 と別の並び）。
  有効画素率が MIN_PCT 未満の観測は保存しない（表示側の下限も同じ）。
  Landsat は 30m 画素が landsat_min_pixels 未満しか入らない小さい区画には使わない。
雲判定の方式（mask）が変わったセルは ndvi_next.json に過去分から計算し直し、
最新まで追いついたら ndvi.json と入れ替える（それまでは古い ndvi.json を表示）。

--backend fake --fake-csv <file> を付けると GEE を使わずに CSV（pid,date,mean,count[,src]）から
同じ形式を作る（動作確認用）。
"""
import argparse, os, sys, json, time, math, datetime as dt
from concurrent.futures import ThreadPoolExecutor, as_completed
from common import load_config, write_json, read_json, round_coords, with_retry

MAX_FEATURES_PER_CALL = 4500      # getInfo の要素数上限(5000)に余裕を持たせる
MIN_PCT = 50                      # これ未満の有効画素率の観測は保存しない
LOOKBACK_DAYS = 15                # 雲判定データの配信遅れに備えて、直近はもう一度確認する
LANDSAT_MASK = "hlsl30-fmask"     # Landsat の雲判定の方式（変えると Landsat 分だけ計算し直す）
RADAR_MASK = "s1-vhdb"            # レーダーの指標（変えるとレーダー分だけ計算し直す）

# 観測元ごとの設定: ndvi.json のキー、計算の画素サイズ、full（10m画素数）との比、直近の再確認日数
SOURCES = {
    "s2": {"dates": "dates", "p": "p", "scale": 10, "div": 1, "lookback": LOOKBACK_DAYS},
    "ls": {"dates": "ldates", "p": "lp", "scale": 30, "div": 9, "lookback": 30},
    # Sentinel-1 レーダー: VH の後方散乱（区画平均を dB に。保存は dB×1000）。梅雨の空白を埋めるため、各年の radar_window の間だけ計算する
    "s1": {"dates": "rdates", "p": "rp", "scale": 10, "div": 1, "lookback": 15},
}


def mask_id(cfg):
    sc = cfg.get("max_scene_cloud")
    return f"csplus{cfg.get('cloud_score_min', 0.6):.2f}" + (f"-scene{sc}" if sc else "")


# ---------------- GEE backend ----------------
class GEEBackend:
    def __init__(self, cfg):
        import ee
        self.ee = ee
        sa = os.environ.get("EE_SERVICE_ACCOUNT"); key = os.environ.get("EE_PRIVATE_KEY"); project = os.environ.get("EE_PROJECT")
        if sa and key:
            creds = ee.ServiceAccountCredentials(sa, key_data=key)
            ee.Initialize(creds, project=project)
        else:                               # ローカルで earthengine authenticate 済みの場合
            ee.Initialize(project=project)
        self.cfg = cfg

    def _col(self, bbox, start, end, src="s2"):
        return {"ls": self._col_ls, "s1": self._col_s1}.get(src, self._col_s2)(bbox, start, end)

    def _col_s1(self, bbox, start, end):
        """Sentinel-1 GRD（IW, VH あり）。雲を通すので雲判定はない。値は VH（線形。区画で平均してから dB にする）。
        田植え直後の水面では低く（-25dB 前後）、稲が茂るほど高くなる。"""
        ee = self.ee
        def prep(img):
            return ee.Image(10).pow(img.select("VH").divide(10)).rename("NDVI").copyProperties(img, ["system:time_start"])
        return (ee.ImageCollection("COPERNICUS/S1_GRD").filterBounds(ee.Geometry.Rectangle(bbox)).filterDate(start, end)
                .filter(ee.Filter.eq("instrumentMode", "IW"))
                .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
                .map(prep))

    def _col_ls(self, bbox, start, end):
        """Landsat 8/9（NASA HLS L30: Sentinel-2 に合わせて補正済み）。
        Fmask の雲・雲の近傍・雲の影・雪、エアロゾル（かすみ）が多い画素は除く。"""
        ee = self.ee
        def prep(img):
            fm = img.select("Fmask")
            ok = fm.bitwiseAnd(0b11110).eq(0).And(fm.rightShift(6).bitwiseAnd(3).neq(3))
            return (img.normalizedDifference(["B5", "B4"]).rename("NDVI").updateMask(ok)
                    .copyProperties(img, ["system:time_start"]))
        return (ee.ImageCollection("NASA/HLS/HLSL30/v002")
                .filterBounds(ee.Geometry.Rectangle(bbox)).filterDate(start, end).map(prep))

    def _col_s2(self, bbox, start, end):
        """雲判定は Cloud Score+（cs_cdf がしきい値以上の画素だけ使う）。
        シーン全体の雲量では捨てない（max_scene_cloud が空欄のとき）ので、晴れ間の区画も拾える。"""
        ee = self.ee
        region = ee.Geometry.Rectangle(bbox)
        thr = self.cfg.get("cloud_score_min", 0.6)
        def prep(img):
            scl = img.select("SCL")
            ok = img.select("cs_cdf").gte(thr).And(scl.neq(0)).And(scl.neq(1)).And(scl.neq(11))   # 欠測・飽和・雪は除く
            return (img.normalizedDifference(["B8", "B4"]).rename("NDVI").updateMask(ok)
                    .copyProperties(img, ["system:time_start"]))
        col = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
               .filterBounds(region).filterDate(start, end))
        if self.cfg.get("max_scene_cloud"):
            col = col.filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", self.cfg["max_scene_cloud"]))
        csp = ee.ImageCollection("GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED").filterBounds(region).filterDate(start, end)
        # 雲判定がまだ出ていない画像は使わない（次回以降、LOOKBACK_DAYS の範囲で拾い直す）
        col = col.filter(ee.Filter.inList("system:index", csp.aggregate_array("system:index")))
        return col.linkCollection(csp, ["cs_cdf"]).map(prep)

    def list_dates(self, bbox, start, end, src="s2"):
        """期間内の観測日（UTC日付, YYYY-MM-DD）を返す"""
        ts = with_retry(lambda: self._col(bbox, start, end, src).aggregate_array("system:time_start").getInfo())
        return sorted({dt.datetime.fromtimestamp(t / 1000, dt.timezone.utc).strftime("%Y-%m-%d") for t in ts})

    def stats(self, bbox, inner_fc_geojson, dates, src="s2"):
        """dates（同日は1枚に合成）ごとの区画平均 → {date: {pid: [mean, count]}}"""
        ee = self.ee
        fc = ee.FeatureCollection(inner_fc_geojson)
        scale = SOURCES[src]["scale"]
        col = self._col(bbox, dates[0], (dt.date.fromisoformat(dates[-1]) + dt.timedelta(days=1)).isoformat(), src)
        imgs = [col.filterDate(d, ee.Date(d).advance(1, "day")).mosaic().set("date", d) for d in dates]
        daily = ee.ImageCollection(imgs)
        reducer = ee.Reducer.mean().combine(ee.Reducer.count(), "", True)
        def per_img(img):
            return img.reduceRegions(collection=fc, reducer=reducer, scale=scale).map(lambda f: f.set("date", img.get("date")))
        table = daily.map(per_img).flatten().filter(ee.Filter.notNull(["mean"]))
        res = with_retry(lambda: table.select(["pid", "date", "mean", "count"], None, False).getInfo())
        out = {d: {} for d in dates}
        for f in res["features"]:
            p = f["properties"]
            out[p["date"]][p["pid"]] = [round(float(p["mean"]), 3), int(p["count"])]
        return out


# ---------------- fake backend（動作確認用） ----------------
class FakeBackend:
    def __init__(self, cfg, csv_path):
        import csv
        self.rows = {}
        with open(csv_path, encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                pid = r.get("pid") or r.get("name"); d = r["date"][:10]; src = r.get("src") or "s2"
                self.rows.setdefault(src, {}).setdefault(d, {}).setdefault(pid, []).append((float(r["mean"]), float(r.get("count") or 1)))
    def list_dates(self, bbox, start, end, src="s2"):
        return sorted(d for d in self.rows.get(src, {}) if start <= d < end)
    def stats(self, bbox, inner_fc_geojson, dates, src="s2"):
        pids = {f["properties"]["pid"] for f in inner_fc_geojson["features"]}
        out = {}
        for d in dates:
            out[d] = {}
            for pid, vals in self.rows.get(src, {}).get(d, {}).items():
                if pid in pids:
                    w = sum(m * c for m, c in vals); c = sum(c for _, c in vals); mx = max(c for _, c in vals)
                    out[d][pid] = [round(w / c, 3), int(mx)]
        return out


# ---------------- main ----------------
def load_full(cdir):
    """pid -> 10m画素の期待数（parcels.geojson の full）"""
    g = read_json(os.path.join(cdir, "parcels.geojson"), {"features": []})
    return {f["properties"]["pid"]: f["properties"].get("full") or 0 for f in g["features"]}


def compact_parcels(cdir):
    """配信用の区画の形（parcels.geojson）の座標を約1m単位に丸めて軽くする（1回だけ）"""
    path = os.path.join(cdir, "parcels.geojson")
    g = read_json(path)
    if not g or g.get("precision") == 5:
        return
    for f in g["features"]:
        f["geometry"]["coordinates"] = round_coords(f["geometry"]["coordinates"])
    g["precision"] = 5
    write_json(path, g)


def sources(cfg):
    return ["s2"] + (["ls"] if cfg.get("landsat") else []) + (["s1"] if cfg.get("radar") else [])


def in_window(cfg, d):
    """レーダーを計算する時期（各年の radar_window、月-日）か"""
    a, b = cfg.get("radar_window", ["05-01", "08-10"])
    return a <= d[5:] <= b


def cell_state(cfg, cdir):
    """(書き込み先, ndvi データ, 並べ替えキー)。方式が変わったセルは ndvi_next.json に計算し直す。"""
    mid = mask_id(cfg)
    cur = read_json(os.path.join(cdir, "ndvi.json"))
    if cur and cur.get("v") == 2 and cur.get("mask") == mid:
        fname, nd = "ndvi.json", cur
    else:
        nxt = read_json(os.path.join(cdir, "ndvi_next.json"))
        if not (nxt and nxt.get("v") == 2 and nxt.get("mask") == mid):
            nxt = {"v": 2, "mask": mid, "dates": [], "p": {}}
        fname, nd = "ndvi_next.json", nxt
    lfresh = "ls" in sources(cfg) and nd.get("lmask") != LANDSAT_MASK
    if lfresh:                                   # Landsat は初回・方式変更時に全期間を計算
        nd.update({"lmask": LANDSAT_MASK, "ldates": [], "lp": {}})
    rfresh = "s1" in sources(cfg) and nd.get("rmask") != RADAR_MASK
    if rfresh:                                   # レーダーも初回・方式変更時に全期間を計算
        nd.update({"rmask": RADAR_MASK, "rdates": [], "rp": {}})
    fresh = fname == "ndvi_next.json" or lfresh or rfresh
    return fname, nd, (" " if fresh else "") + (nd["dates"][-1] if nd["dates"] else "")   # 計算し直しのセルを先に


def sort_dates(nd, src="s2"):
    """dates を昇順に並べ直し、区画ごとの添字も付け替える"""
    kd, kp = SOURCES[src]["dates"], SOURCES[src]["p"]
    if nd[kd] == sorted(nd[kd]):
        return
    order = sorted(range(len(nd[kd])), key=lambda i: nd[kd][i])
    new_i = {old: new for new, old in enumerate(order)}
    nd[kd] = [nd[kd][i] for i in order]
    for pid, a in nd[kp].items():
        t = sorted(((new_i[a[k]], a[k + 1], a[k + 2]) for k in range(0, len(a), 3)))
        nd[kp][pid] = [x for tr in t for x in tr]


def update_source(backend, cfg, cdir, fname, nd, cell, inner, full, src, start_default, end):
    """1つの観測元について、まだ計算していない観測日を足す。足した日数を返す。"""
    S = SOURCES[src]; kd, kp = S["dates"], S["p"]
    min_px = cfg.get("landsat_min_pixels", 2) if src == "ls" else 1
    if src == "ls":                              # 30m画素が十分入る区画だけ計算する
        inner = {"type": "FeatureCollection", "features": [f for f in inner["features"]
                 if (full.get(f["properties"]["pid"]) or 0) / S["div"] >= min_px]}
        if not inner["features"]:
            return 0
    pids = [f["properties"]["pid"] for f in inner["features"]]
    have = set(nd[kd])
    start = start_default
    if nd[kd]:
        nxt = (dt.date.fromisoformat(nd[kd][-1]) + dt.timedelta(days=1)).isoformat()
        back = (dt.date.fromisoformat(end) - dt.timedelta(days=S["lookback"])).isoformat()
        start = max(start_default, min(nxt, back))
    dates = [d for d in backend.list_dates(cell["bbox"], start, end, src) if d not in have and (src != "s1" or in_window(cfg, d))]
    added = 0
    per_call = max(1, MAX_FEATURES_PER_CALL // max(1, len(pids)))
    for i in range(0, len(dates), per_call):
        chunk = dates[i:i + per_call]
        res = backend.stats(cell["bbox"], inner, chunk, src)
        for d in chunk:
            di = len(nd[kd]); nd[kd].append(d)
            for pid in pids:
                v = res.get(d, {}).get(pid)
                if not v or v[1] < min_px:
                    continue
                f = (full.get(pid) or 0) / S["div"] or v[1] or 1
                pct = int(100 * min(1.0, v[1] / f) + 1e-6)   # 切り捨て（しきい値の判定を従来と同じにする）
                if pct >= MIN_PCT:
                    val = 10 * math.log10(max(v[0], 1e-6)) if src == "s1" else v[0]   # レーダーは dB
                    nd[kp].setdefault(pid, []).extend([di, round(val * 1000), pct])
            added += 1
        sort_dates(nd, src)
        # 途中で落ちても進捗を失わないよう都度保存
        write_json(os.path.join(cdir, fname), nd)
    return added


def process_cell(backend, cfg, data_dir, cell, start_default, end, log):
    cid = cell["id"]; cdir = os.path.join(data_dir, "cells", cid)
    inner = read_json(os.path.join(cdir, "inner.geojson"))
    if not inner:
        return cid, 0
    fname, nd, _ = cell_state(cfg, cdir)
    full = load_full(cdir)
    got = {src: update_source(backend, cfg, cdir, fname, nd, cell, inner, full, src, start_default, end) for src in sources(cfg)}
    added = sum(got.values())
    npar = len(inner["features"])
    if fname == "ndvi_next.json":              # 最新まで追いついたので入れ替える
        write_json(os.path.join(cdir, "ndvi.json"), nd)
        if os.path.exists(os.path.join(cdir, "ndvi_next.json")):
            os.remove(os.path.join(cdir, "ndvi_next.json"))
        log(f"{cid}: 新しい雲判定で計算し直し完了 ({len(nd['dates'])}日, 区画{npar})")
    elif added:
        log(f"{cid}: " + ", ".join(f"{k} +{n}日" for k, n in got.items()) + f" (区画{npar})")
    return cid, added


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-minutes", type=float, default=300)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--backend", choices=["gee", "fake"], default="gee")
    ap.add_argument("--fake-csv")
    ap.add_argument("--cells", help="カンマ区切りでセルIDを限定（テスト用）")
    ap.add_argument("--pixel-minutes", type=float, default=240, help="圃場内マップの画素データ作成に使う時間の上限（分）")
    args = ap.parse_args()

    cfg = load_config()
    data_dir = os.path.join(cfg["site_dir"], "data")
    index = read_json(os.path.join(data_dir, "index.json"))
    if not index:
        sys.exit("index.json がありません。先に build_cells.py を実行してください。")
    backend = FakeBackend(cfg, args.fake_csv) if args.backend == "fake" else GEEBackend(cfg)

    today = dt.date.today()
    end = (today + dt.timedelta(days=1)).isoformat()
    start_default = (today - dt.timedelta(days=30 * cfg["history_months"])).isoformat()
    for c in index["cells"]:
        compact_parcels(os.path.join(data_dir, "cells", c["id"]))
    cells = index["cells"]
    if args.cells:
        want = set(args.cells.split(",")); cells = [c for c in cells if c["id"] in want]
    # 更新が古いセルから処理（時間切れでも次回に続きができる）
    cells = sorted(cells, key=lambda c: cell_state(cfg, os.path.join(data_dir, "cells", c["id"]))[2])

    t0 = time.time(); total = 0; done = 0
    def log(m): print(time.strftime("%H:%M:%S"), m, flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {}
        it = iter(cells)
        # 時間予算を見ながら順次投入
        pending = list(cells)
        while pending or futs:
            while pending and len(futs) < args.workers and (time.time() - t0) < args.max_minutes * 60:
                c = pending.pop(0)
                futs[ex.submit(process_cell, backend, cfg, data_dir, c, start_default, end, log)] = c["id"]
            if not futs:
                break
            for f in as_completed(list(futs)):
                cid = futs.pop(f)
                try:
                    _, n = f.result(); total += n; done += 1
                except Exception as e:      # 1セルの失敗で全体を止めない
                    log(f"{cid}: エラー {e}")
                break
            if (time.time() - t0) >= args.max_minutes * 60 and pending:
                log(f"時間切れ。残り {len(pending)} セルは次回に続けます。"); pending = []
    log(f"NDVI 完了: {done}セル処理, 観測日を合計 {total} 追加")

    # 圃場内マップ（10m画素）。NDVI のあとに残り時間で作る。途中で切れても次回に続きから。
    if cfg.get("pixel_maps") and args.backend == "gee":
        from pixels import GEEPixels, update_cell, windows
        psrc = GEEPixels(backend); mid = mask_id(cfg)
        t1 = time.time(); limit = min(args.pixel_minutes * 60, 330 * 60 - (t1 - t0))
        cur = windows(cfg, today)[-1][0]
        def px_key(c):          # 今年の分がまだないセルから
            m = read_json(os.path.join(data_dir, "cells", c["id"], f"px_{cur}.json")) or {}
            return (m.get("mask") == mid, max(m.get("src_dates") or [""]))
        pcells = sorted(cells, key=px_key); pdone = 0; pdays = 0
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {}; pending = list(pcells)
            while pending or futs:
                while pending and len(futs) < args.workers and time.time() - t1 < limit:
                    c = pending.pop(0)
                    futs[ex.submit(update_cell, psrc, cfg, data_dir, c, today, mid, log)] = c["id"]
                if not futs:
                    break
                for f in as_completed(list(futs)):
                    cid = futs.pop(f)
                    try:
                        pdays += f.result(); pdone += 1
                    except Exception as e:
                        log(f"{cid}: 圃場内マップ エラー {e}")
                    break
                if time.time() - t1 >= limit and pending:
                    log(f"圃場内マップ 時間切れ。残り {len(pending)} セルは次回に続けます。"); pending = []
        log(f"圃場内マップ 完了: {pdone}セル, 画素データを合計 {pdays} 日分取得")

    index["updated"] = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    index["source"] = "Sentinel-2 L2A (Copernicus)" + (" + Landsat 8/9 (NASA HLS)" if cfg.get("landsat") else "") + (" + Sentinel-1" if cfg.get("radar") else "") + " / Google Earth Engine"
    write_json(os.path.join(data_dir, "index.json"), index)
    log(f"完了: {done}セル処理, 観測日を合計 {total} 追加")

if __name__ == "__main__":
    main()
