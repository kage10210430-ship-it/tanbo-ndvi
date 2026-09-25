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
RADAR_MASK = "s1-vhdb-med"        # レーダーの指標（変えるとレーダー分だけ計算し直す）。区画の中央値（縁の畦・道路・建物に引っぱられない）
RECENT_DAYS = 20                  # 毎日の更新で、新しい画像を探す日数
PROVISIONAL_DAYS = 7              # この日数以内の日は「仮」。同じ日の画像があとから届いたら計算し直す
SEEN_KEEP_DAYS = 40               # state.json に覚えておく画像の日数

# 観測元ごとの設定: ndvi.json のキー、計算の画素サイズ、full（10m画素数）との比、直近の再確認日数
SOURCES = {
    "s2": {"dates": "dates", "p": "p", "scale": 10, "div": 1, "lookback": LOOKBACK_DAYS},
    "ls": {"dates": "ldates", "p": "lp", "scale": 30, "div": 9, "lookback": 30},
    # Sentinel-1 レーダー: VH の後方散乱（区画の中央値を dB に。保存は dB×1000）。水張り・梅雨の空白を見るため、各年の radar_window の間だけ計算する
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
        """NDVI（レーダーは VH）1バンドの画像の集まり"""
        raw = self._raw(bbox, start, end, src)
        return raw.linkCollection(self._csp(bbox, start, end), ["cs_cdf"]).map(self._prep_s2) if src == "s2" else raw.map(
            {"ls": self._prep_ls, "s1": self._prep_s1}[src])

    def _raw(self, bbox, start, end, src="s2"):
        """加工前の画像の集まり（画像のID・写っている範囲を調べるのにも使う）"""
        ee = self.ee
        region = ee.Geometry.Rectangle(bbox)
        if src == "s1":
            # Sentinel-1 GRD（IW, VH あり）。雲を通すので雲判定はない
            return (ee.ImageCollection("COPERNICUS/S1_GRD").filterBounds(region).filterDate(start, end)
                    .filter(ee.Filter.eq("instrumentMode", "IW"))
                    .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH")))
        if src == "ls":
            # Landsat 8/9（NASA HLS L30: Sentinel-2 に合わせて補正済み）
            return ee.ImageCollection("NASA/HLS/HLSL30/v002").filterBounds(region).filterDate(start, end)
        col = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(region).filterDate(start, end)
        if self.cfg.get("max_scene_cloud"):
            col = col.filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", self.cfg["max_scene_cloud"]))
        # 雲判定（Cloud Score+）がまだ出ていない画像は使わない（出たときに「新しい画像」として取り込む）
        return col.filter(ee.Filter.inList("system:index", self._csp(bbox, start, end).aggregate_array("system:index")))

    def _csp(self, bbox, start, end):
        ee = self.ee
        return ee.ImageCollection("GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED").filterBounds(ee.Geometry.Rectangle(bbox)).filterDate(start, end)

    def _prep_s1(self, img):
        """値は VH（線形。区画の中央値をとってから dB にする）。田植え直後の水面では低く、稲が茂るほど高くなる"""
        ee = self.ee
        return ee.Image(10).pow(img.select("VH").divide(10)).rename("NDVI").copyProperties(img, ["system:time_start"])

    def _prep_ls(self, img):
        """Fmask の雲・雲の近傍・雲の影・雪、エアロゾル（かすみ）が多い画素は除く"""
        fm = img.select("Fmask")
        ok = fm.bitwiseAnd(0b11110).eq(0).And(fm.rightShift(6).bitwiseAnd(3).neq(3))
        return (img.normalizedDifference(["B5", "B4"]).rename("NDVI").updateMask(ok)
                .copyProperties(img, ["system:time_start"]))

    def _prep_s2(self, img):
        """雲判定は Cloud Score+（cs_cdf がしきい値以上の画素だけ使う）。欠測・飽和・雪（SCL 0/1/11）も除く。
        シーン全体の雲量では捨てない（max_scene_cloud が空欄のとき）ので、晴れ間の区画も拾える。"""
        scl = img.select("SCL")
        ok = img.select("cs_cdf").gte(self.cfg.get("cloud_score_min", 0.6)).And(scl.neq(0)).And(scl.neq(1)).And(scl.neq(11))
        return (img.normalizedDifference(["B8", "B4"]).rename("NDVI").updateMask(ok)
                .copyProperties(img, ["system:time_start"]))

    def day(self, col, d):
        """その日の画像を1枚に合成。画像が1枚もない日でも、全部マスクされた NDVI バンドを返す（エラーにしない）"""
        ee = self.ee
        base = ee.Image.constant(0).toFloat().rename("NDVI").updateMask(0)
        return ee.ImageCollection([base]).merge(col.filterDate(d, ee.Date(d).advance(1, "day"))).mosaic()

    def images(self, bbox, start, end, src="s2"):
        """期間内の画像の一覧 [{id, t(ミリ秒), geom(写っている範囲 GeoJSON)}]。毎日の更新で新しい画像を探すのに使う"""
        ee = self.ee
        fc = self._raw(bbox, start, end, src).map(
            # 写っている範囲は平面（経緯度の直線）で 10m 精度に細かくして返す。手元の shapely の判定と GEE の判定をそろえるため
            lambda i: ee.Feature(i.geometry(10, "EPSG:4326", False), {"id": i.get("system:index"), "t": i.get("system:time_start")}))
        info = with_retry(lambda: fc.getInfo())
        return [{"id": f["properties"]["id"], "t": f["properties"]["t"], "geom": f["geometry"]} for f in info["features"]]

    def list_dates(self, bbox, start, end, src="s2"):
        """期間内の観測日（UTC日付, YYYY-MM-DD）を返す"""
        ts = with_retry(lambda: self._col(bbox, start, end, src).aggregate_array("system:time_start").getInfo())
        return sorted({dt.datetime.fromtimestamp(t / 1000, dt.timezone.utc).strftime("%Y-%m-%d") for t in ts})

    def stats(self, bbox, inner_fc_geojson, dates, src="s2"):
        """dates（同日は1枚に合成）ごとの区画平均（レーダーは中央値）→ {date: {pid: [値, count]}}"""
        ee = self.ee
        fc = ee.FeatureCollection(inner_fc_geojson)
        scale = SOURCES[src]["scale"]
        col = self._col(bbox, dates[0], (dt.date.fromisoformat(dates[-1]) + dt.timedelta(days=1)).isoformat(), src)
        imgs = [self.day(col, d).set("date", d) for d in dates]
        daily = ee.ImageCollection(imgs)
        stat = "median" if src == "s1" else "mean"
        reducer = (ee.Reducer.median() if src == "s1" else ee.Reducer.mean()).combine(ee.Reducer.count(), "", True)
        def per_img(img):
            return img.reduceRegions(collection=fc, reducer=reducer, scale=scale).map(lambda f: f.set("date", img.get("date")))
        table = daily.map(per_img).flatten().filter(ee.Filter.notNull([stat]))
        res = with_retry(lambda: table.select(["pid", "date", stat, "count"], None, False).getInfo())
        out = {d: {} for d in dates}
        for f in res["features"]:
            p = f["properties"]
            # レーダーの線形の値は 0.001〜0.02 ほどなので、小数3桁に丸めると dB が −20.0/−20.5/−21.0… のとびとびになる。桁を残す
            out[p["date"]][p["pid"]] = [round(float(p[stat]), 7 if src == "s1" else 3), int(p["count"])]
        return out


# ---------------- fake backend（動作確認用） ----------------
class FakeBackend:
    def __init__(self, cfg, csv_path):
        import csv
        self.rows = {}; self.imgs = {}
        with open(csv_path, encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                pid = r.get("pid") or r.get("name"); d = r["date"][:10]; src = r.get("src") or "s2"
                self.rows.setdefault(src, {}).setdefault(d, {}).setdefault(pid, []).append((float(r["mean"]), float(r.get("count") or 1)))
                self.imgs.setdefault(src, {}).setdefault(d, set()).add(r.get("img") or f"{src}_{d}")
    def list_dates(self, bbox, start, end, src="s2"):
        return sorted(d for d in self.rows.get(src, {}) if start <= d < end)
    def images(self, bbox, start, end, src="s2"):          # 写っている範囲は県全体とみなす（geom なし）
        return [{"id": i, "t": int(dt.datetime.fromisoformat(d + "T01:30:00+00:00").timestamp() * 1000), "geom": None}
                for d, ids in self.imgs.get(src, {}).items() if start <= d < end for i in sorted(ids)]
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


def drop_dates(nd, src, drop):
    """指定した日の値を消して添字を詰める（あとから画像が届いた日を計算し直すため）。消した日数を返す"""
    kd, kp = SOURCES[src]["dates"], SOURCES[src]["p"]
    drop = set(drop) & set(nd.get(kd, []))
    if not drop:
        return 0
    keep = [i for i, d in enumerate(nd[kd]) if d not in drop]
    new_i = {old: new for new, old in enumerate(keep)}
    nd[kd] = [nd[kd][i] for i in keep]
    for pid in list(nd[kp]):
        a = nd[kp][pid]
        t = [x for k in range(0, len(a), 3) if a[k] in new_i for x in (new_i[a[k]], a[k + 1], a[k + 2])]
        if t:
            nd[kp][pid] = t
        else:
            del nd[kp][pid]
    return len(drop)


def update_source(backend, cfg, cdir, fname, nd, cell, inner, full, src, start_default, end, only_dates=None):
    """1つの観測元について、まだ計算していない観測日を足す。足した日数を返す。
    only_dates を渡すと（毎日の更新）、観測日を調べ直さずにその日だけ計算する。"""
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
    cand = sorted(d for d in only_dates if start_default <= d < end) if only_dates is not None else backend.list_dates(cell["bbox"], start, end, src)
    dates = [d for d in cand if d not in have and (src != "s1" or in_window(cfg, d))]
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


def process_cell(backend, cfg, data_dir, cell, start_default, end, log, plan=None, targeted=False):
    """plan = {src: {"dates": 新しい画像がある日, "redo": 計算し直す日}}。
    targeted（毎日の更新）なら plan の日だけ計算する。そうでなければ観測日を調べて足りない日をすべて計算する。"""
    cid = cell["id"]; cdir = os.path.join(data_dir, "cells", cid)
    inner = read_json(os.path.join(cdir, "inner.geojson"))
    if not inner:
        return cid, 0
    fname, nd, key = cell_state(cfg, cdir)
    if key.startswith(" "):
        targeted = False                        # 計算し直し中のセルは全部調べる
    plan = plan or {}
    redone = 0
    full = load_full(cdir)
    got = {}
    for src in sources(cfg):
        if targeted and src not in plan:
            continue
        # あとから画像が届いた「仮」の日は消して計算し直す。消したことは計算結果と一緒に保存する
        # （計算が失敗したら保存されないので、前の値が残る）
        if SOURCES[src]["dates"] in nd:
            redone += drop_dates(nd, src, plan.get(src, {}).get("redo", ()))
        got[src] = update_source(backend, cfg, cdir, fname, nd, cell, inner, full, src, start_default, end,
                                 only_dates=sorted(plan[src]["dates"]) if targeted else None)
    added = sum(got.values())
    npar = len(inner["features"])
    if fname == "ndvi_next.json":              # 最新まで追いついたので入れ替える
        write_json(os.path.join(cdir, "ndvi.json"), nd)
        if os.path.exists(os.path.join(cdir, "ndvi_next.json")):
            os.remove(os.path.join(cdir, "ndvi_next.json"))
        log(f"{cid}: 新しい雲判定で計算し直し完了 ({len(nd['dates'])}日, 区画{npar})")
    elif added:
        log(f"{cid}: " + ", ".join(f"{k} +{n}日" for k, n in got.items()) + f" (区画{npar})")
    last = {src: max(nd[SOURCES[src]["dates"]]) for src in got if nd.get(SOURCES[src]["dates"])}
    return cid, added, last


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-minutes", type=float, default=300)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--backend", choices=["gee", "fake"], default="gee")
    ap.add_argument("--fake-csv")
    ap.add_argument("--cells", help="カンマ区切りでセルIDを限定（テスト用）")
    ap.add_argument("--pixel-minutes", type=float, default=240, help="圃場内マップの画素データ作成に使う時間の上限（分）")
    ap.add_argument("--full", action="store_true", help="新しい画像の有無に関係なく全セルを調べる")
    args = ap.parse_args()

    cfg = load_config()
    data_dir = os.path.join(cfg["site_dir"], "data")
    index = read_json(os.path.join(data_dir, "index.json"))
    if not index:
        sys.exit("index.json がありません。先に build_cells.py を実行してください。")
    backend = FakeBackend(cfg, args.fake_csv) if args.backend == "fake" else GEEBackend(cfg)

    now = dt.datetime.now(dt.timezone.utc)
    today = now.date()
    end = (today + dt.timedelta(days=1)).isoformat()
    start_default = (today - dt.timedelta(days=30 * cfg["history_months"])).isoformat()
    t0 = time.time()
    def log(m): print(time.strftime("%H:%M:%S"), m, flush=True)
    for c in index["cells"]:
        compact_parcels(os.path.join(data_dir, "cells", c["id"]))

    # ---- 新しい画像を県全体で探す（毎日の更新）----
    state_p = os.path.join(data_dir, "state.json")
    state = read_json(state_p, {}) or {}
    seen = state.setdefault("seen", {})          # {src: {画像ID: {date, t, first_seen}}}
    plans, found_all = find_new_images(backend, cfg, index, seen, today, end, now, log)
    # 前回やり残したセル（時間切れ・エラー）の予定を足す。成功するまで毎回やり直す
    for cid, pl in state.get("pending", {}).items():
        for src, v in pl.items():
            p = plans.setdefault(cid, {}).setdefault(src, {"dates": set(), "redo": set()})
            p["dates"] |= set(v.get("dates", [])); p["redo"] |= set(v.get("redo", []))
    px_pending = state.get("px_pending", {})       # 圃場内マップのやり残し {セルID: 作り直す日}
    redo_full = set(state.get("pending_full", []))  # 全セルを調べる更新でやり残したセル（次回も全部調べる）
    jst_monday = (now + dt.timedelta(hours=9)).weekday() == 0
    full = args.full or not found_all or jst_monday
    log("全セルを調べる更新（週1回・初回・指定時）" if full else f"毎日の更新: 新しい画像があるセル {len(plans)}")

    cells = index["cells"]
    if args.cells:
        want = set(args.cells.split(",")); cells = [c for c in cells if c["id"] in want]
    if not full:                                  # 新しい画像があるセルと、計算し直し中のセルだけ
        cells = [c for c in cells if c["id"] in plans or c["id"] in redo_full
                 or cell_state(cfg, os.path.join(data_dir, "cells", c["id"]))[2].startswith(" ")]
    # 新しい画像があるセルを先に、次に更新が古いセル（計算し直し中のセルが時間を使い切って、新しい画像が後回しにならないように）
    cells = sorted(cells, key=lambda c: (c["id"] not in plans and c["id"] not in redo_full, cell_state(cfg, os.path.join(data_dir, "cells", c["id"]))[2]))

    total = 0; done = 0; changed = False; ok_cells = set(); latest_new = {}
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {}
        # 時間予算を見ながら順次投入
        pending = list(cells)
        while pending or futs:
            while pending and len(futs) < args.workers and (time.time() - t0) < args.max_minutes * 60:
                c = pending.pop(0)
                futs[ex.submit(process_cell, backend, cfg, data_dir, c, start_default, end, log,
                               plans.get(c["id"]), not full and c["id"] not in redo_full)] = c["id"]
            if not futs:
                break
            for f in as_completed(list(futs)):
                cid = futs.pop(f)
                try:
                    _, n, last = f.result(); total += n; done += 1; ok_cells.add(cid)
                    changed = changed or n > 0 or any(v.get("redo") for v in plans.get(cid, {}).values())
                    for s_, d_ in last.items():
                        latest_new[s_] = max(latest_new.get(s_, ""), d_)
                except Exception as e:      # 1セルの失敗で全体を止めない（予定は次回に持ち越す）
                    log(f"{cid}: エラー {e}")
                break
            if (time.time() - t0) >= args.max_minutes * 60 and pending:
                log(f"時間切れ。残り {len(pending)} セルは次回に続けます。"); pending = []
    state["pending"] = {cid: {src: {"dates": sorted(v["dates"]), "redo": sorted(v["redo"])} for src, v in pl.items()}
                        for cid, pl in plans.items() if cid not in ok_cells}
    state["pending_full"] = sorted({c["id"] for c in cells if c["id"] not in ok_cells and (full or c["id"] in redo_full)})
    if state["pending"] or state["pending_full"]:
        log(f"やり残し {len(set(state['pending']) | set(state['pending_full']))} セルは次回に続けます")
    log(f"NDVI 完了: {done}セル処理, 観測日を合計 {total} 追加")

    # 圃場内マップ（10m画素）。NDVI のあとに残り時間で作る。途中で切れても次回に続きから。
    pdays = 0
    if cfg.get("pixel_maps") and args.backend == "gee":
        from pixels import GEEPixels, update_cell, windows, drop_dates as px_drop
        psrc = GEEPixels(backend); mid = mask_id(cfg)
        t1 = time.time(); limit = min(args.pixel_minutes * 60, 330 * 60 - (t1 - t0))
        cur = windows(cfg, today)[-1][0]
        # 圃場内マップを作るセル: NDVI が成功したセル（毎日の更新では S2 の新しい画像があったセル）と、前回のやり残し
        px_redo = {cid: set(v) for cid, v in px_pending.items()}
        for cid in ok_cells:
            r = plans.get(cid, {}).get("s2", {}).get("redo")
            if r:
                px_redo.setdefault(cid, set()).update(r)
        pc_ids = {c["id"] for c in cells if c["id"] in ok_cells and (full or "s2" in plans.get(c["id"], {}))} | set(px_pending)
        pcells = [c for c in index["cells"] if c["id"] in pc_ids]
        def px_task(c):                           # あとから画像が届いた「仮」の日は、作る直前に画素も消して作り直す
            if px_redo.get(c["id"]):
                px_drop(os.path.join(data_dir, "cells", c["id"]), px_redo[c["id"]])
            return update_cell(psrc, cfg, data_dir, c, today, mid, log)
        def px_key(c):          # 今年の分がまだないセルから
            m = read_json(os.path.join(data_dir, "cells", c["id"], f"px_{cur}.json")) or {}
            return (m.get("mask") == mid, max(m.get("src_dates") or [""]))
        pcells = sorted(pcells, key=px_key); pdone = 0; px_ok = set()
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {}; pending = list(pcells)
            while pending or futs:
                while pending and len(futs) < args.workers and time.time() - t1 < limit:
                    c = pending.pop(0)
                    futs[ex.submit(px_task, c)] = c["id"]
                if not futs:
                    break
                for f in as_completed(list(futs)):
                    cid = futs.pop(f)
                    try:
                        pdays += f.result(); pdone += 1; px_ok.add(cid)
                    except Exception as e:
                        log(f"{cid}: 圃場内マップ エラー {e}")
                    break
                if time.time() - t1 >= limit and pending:
                    log(f"圃場内マップ 時間切れ。残り {len(pending)} セルは次回に続けます。"); pending = []
        state["px_pending"] = {cid: sorted(px_redo.get(cid, ())) for cid in pc_ids if cid not in px_ok}
        log(f"圃場内マップ 完了: {pdone}セル, 画素データを合計 {pdays} 日分取得")

    stamp = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    index["checked"] = stamp                       # 最後に新しい画像を確認した時刻
    if changed or pdays or "updated" not in index:
        index["updated"] = stamp                   # データが変わった時刻
    latest = index.setdefault("latest", {})        # 観測元ごとの最新の撮影日（実際に計算したデータの中で）
    for src, d in latest_new.items():
        latest[src] = max(latest.get(src, ""), d)
    index["source"] = "Sentinel-2 L2A (Copernicus)" + (" + Landsat 8/9 (NASA HLS)" if cfg.get("landsat") else "") + (" + Sentinel-1" if cfg.get("radar") else "") + " / Google Earth Engine"
    write_json(os.path.join(data_dir, "index.json"), index)
    write_json(state_p, state)
    log(f"完了: {done}セル処理, 観測日を合計 {total} 追加（{(time.time() - t0) / 60:.1f}分）")


def find_new_images(backend, cfg, index, seen, today, end, now, log):
    """県全体で直近 RECENT_DAYS 日の画像を調べ、まだ見ていない画像が写っているセルを返す。
    戻り値: ({セルID: {src: {"dates": set, "redo": set}}}, すべての観測元で調べられたか)。
    seen は更新する（初めて見た時刻も記録し、撮影から GEE に入るまでの遅れを測れるようにする）。"""
    from shapely.geometry import shape, box
    from shapely.prepared import prep
    since = (today - dt.timedelta(days=RECENT_DAYS)).isoformat()
    cutoff = (today - dt.timedelta(days=SEEN_KEEP_DAYS)).isoformat()
    first_run = not any(seen.values())
    boxes = [(c["id"], box(*c["bbox"])) for c in index["cells"]]
    plans = {}; ok = True; stamp = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    for src in sources(cfg):
        ss = seen.setdefault(src, {})
        try:
            imgs = backend.images(index["bbox"], since, end, src)
        except Exception as e:
            log(f"{src}: 新しい画像の確認に失敗（全セルを調べます） {e}"); ok = False; continue
        new = []
        for im in imgs:
            d = dt.datetime.fromtimestamp(im["t"] / 1000, dt.timezone.utc).strftime("%Y-%m-%d")
            if im["id"] in ss or (src == "s1" and not in_window(cfg, d)):
                continue
            ss[im["id"]] = {"date": d, "t": im["t"], "first_seen": stamp}
            new.append((im, d))
        for k in [k for k, v in ss.items() if v["date"] < cutoff]:
            del ss[k]
        if new and not first_run:
            lag = sorted((now.timestamp() * 1000 - im["t"]) / 864e5 for im, _ in new)
            log(f"{src}: 新しい画像 {len(new)}枚（撮影から見つかるまで {lag[0]:.1f}〜{lag[-1]:.1f}日, 中央 {lag[len(lag) // 2]:.1f}日）")
        for im, d in new:
            g = prep(shape(im["geom"])) if im.get("geom") else None
            redo = (today - dt.date.fromisoformat(d)).days <= PROVISIONAL_DAYS
            for cid, b in boxes:
                if g is None or g.intersects(b):
                    pl = plans.setdefault(cid, {}).setdefault(src, {"dates": set(), "redo": set()})
                    pl["dates"].add(d)
                    if redo:
                        pl["redo"].add(d)
    return plans, ok and not first_run


if __name__ == "__main__":
    main()
