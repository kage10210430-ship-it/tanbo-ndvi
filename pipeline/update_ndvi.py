"""各セルの区画について、まだ計算していない観測日の Sentinel-2 NDVI（区画平均）を
Google Earth Engine で計算し、cells/<id>/ndvi.json に追記する。

  python pipeline/update_ndvi.py --max-minutes 300 --workers 4

ndvi.json の形式（v2）:
  {"v": 2, "mask": "<雲判定の方式>",
   "dates": ["2025-09-25", ...],                     # 計算済みの観測日（昇順）
   "p": {"<pid>": [i, ndvi1000, pct, i, ndvi1000, pct, ...]}}
    i = dates の添字, ndvi1000 = NDVI×1000 の整数, pct = 有効画素率(%)
  有効画素率が MIN_PCT 未満の観測は保存しない（表示側の下限も同じ）。
雲判定の方式（mask）が変わったセルは ndvi_next.json に過去分から計算し直し、
最新まで追いついたら ndvi.json と入れ替える（それまでは古い ndvi.json を表示）。

--backend fake --fake-csv <file> を付けると GEE を使わずに CSV（pid,date,mean,count）から
同じ形式を作る（動作確認用）。
"""
import argparse, os, sys, json, time, random, datetime as dt
from concurrent.futures import ThreadPoolExecutor, as_completed
from common import load_config, write_json, read_json

MAX_FEATURES_PER_CALL = 4500      # getInfo の要素数上限(5000)に余裕を持たせる
MIN_PCT = 50                      # これ未満の有効画素率の観測は保存しない
LOOKBACK_DAYS = 15                # 雲判定データの配信遅れに備えて、直近はもう一度確認する


def with_retry(fn, tries=6):
    """GEE の同時実行数制限（Too many concurrent aggregations など）は待って再試行する"""
    for k in range(tries):
        try:
            return fn()
        except Exception as e:
            msg = str(e)
            if k == tries - 1 or not any(x in msg for x in ("Too many concurrent", "Too Many Requests", "429", "rate limit")):
                raise
            time.sleep(15 * 2 ** k + random.uniform(0, 10))


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

    def _col(self, bbox, start, end):
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

    def list_dates(self, bbox, start, end):
        """期間内の観測日（UTC日付, YYYY-MM-DD）を返す"""
        ts = with_retry(lambda: self._col(bbox, start, end).aggregate_array("system:time_start").getInfo())
        return sorted({dt.datetime.fromtimestamp(t / 1000, dt.timezone.utc).strftime("%Y-%m-%d") for t in ts})

    def stats(self, bbox, inner_fc_geojson, dates):
        """dates（同日は1枚に合成）ごとの区画平均 → {date: {pid: [mean, count]}}"""
        ee = self.ee
        fc = ee.FeatureCollection(inner_fc_geojson)
        col = self._col(bbox, dates[0], (dt.date.fromisoformat(dates[-1]) + dt.timedelta(days=1)).isoformat())
        imgs = [col.filterDate(d, ee.Date(d).advance(1, "day")).mosaic().set("date", d) for d in dates]
        daily = ee.ImageCollection(imgs)
        reducer = ee.Reducer.mean().combine(ee.Reducer.count(), "", True)
        def per_img(img):
            return img.reduceRegions(collection=fc, reducer=reducer, scale=10).map(lambda f: f.set("date", img.get("date")))
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
                pid = r.get("pid") or r.get("name"); d = r["date"][:10]
                self.rows.setdefault(d, {}).setdefault(pid, []).append((float(r["mean"]), float(r.get("count") or 1)))
    def list_dates(self, bbox, start, end):
        return sorted(d for d in self.rows if start <= d < end)
    def stats(self, bbox, inner_fc_geojson, dates):
        pids = {f["properties"]["pid"] for f in inner_fc_geojson["features"]}
        out = {}
        for d in dates:
            out[d] = {}
            for pid, vals in self.rows.get(d, {}).items():
                if pid in pids:
                    w = sum(m * c for m, c in vals); c = sum(c for _, c in vals); mx = max(c for _, c in vals)
                    out[d][pid] = [round(w / c, 3), int(mx)]
        return out


# ---------------- main ----------------
def load_full(cdir):
    """pid -> 10m画素の期待数（parcels.geojson の full）"""
    g = read_json(os.path.join(cdir, "parcels.geojson"), {"features": []})
    return {f["properties"]["pid"]: f["properties"].get("full") or 0 for f in g["features"]}


def cell_state(cfg, cdir):
    """(書き込み先, ndvi データ, 並べ替えキー)。方式が変わったセルは ndvi_next.json に計算し直す。"""
    mid = mask_id(cfg)
    cur = read_json(os.path.join(cdir, "ndvi.json"))
    if cur and cur.get("v") == 2 and cur.get("mask") == mid:
        return "ndvi.json", cur, cur["dates"][-1] if cur["dates"] else ""
    nxt = read_json(os.path.join(cdir, "ndvi_next.json"))
    if not (nxt and nxt.get("v") == 2 and nxt.get("mask") == mid):
        nxt = {"v": 2, "mask": mid, "dates": [], "p": {}}
    return "ndvi_next.json", nxt, " " + (nxt["dates"][-1] if nxt["dates"] else "")   # 計算し直しのセルを先に


def sort_dates(nd):
    """dates を昇順に並べ直し、区画ごとの添字も付け替える"""
    if nd["dates"] == sorted(nd["dates"]):
        return
    order = sorted(range(len(nd["dates"])), key=lambda i: nd["dates"][i])
    new_i = {old: new for new, old in enumerate(order)}
    nd["dates"] = [nd["dates"][i] for i in order]
    for pid, a in nd["p"].items():
        t = sorted(((new_i[a[k]], a[k + 1], a[k + 2]) for k in range(0, len(a), 3)))
        nd["p"][pid] = [x for tr in t for x in tr]


def process_cell(backend, cfg, data_dir, cell, start_default, end, log):
    cid = cell["id"]; cdir = os.path.join(data_dir, "cells", cid)
    inner = read_json(os.path.join(cdir, "inner.geojson"))
    if not inner:
        return cid, 0
    fname, nd, _ = cell_state(cfg, cdir)
    full = load_full(cdir)
    pids = [f["properties"]["pid"] for f in inner["features"]]
    have = set(nd["dates"])
    start = start_default
    if nd["dates"]:
        nxt = (dt.date.fromisoformat(nd["dates"][-1]) + dt.timedelta(days=1)).isoformat()
        back = (dt.date.fromisoformat(end) - dt.timedelta(days=LOOKBACK_DAYS)).isoformat()
        start = max(start_default, min(nxt, back))
    dates = [d for d in backend.list_dates(cell["bbox"], start, end) if d not in have]
    added = 0
    per_call = max(1, MAX_FEATURES_PER_CALL // max(1, len(pids)))
    for i in range(0, len(dates), per_call):
        chunk = dates[i:i + per_call]
        res = backend.stats(cell["bbox"], inner, chunk)
        for d in chunk:
            di = len(nd["dates"]); nd["dates"].append(d)
            for pid in pids:
                v = res.get(d, {}).get(pid)
                if not v:
                    continue
                f = full.get(pid) or v[1] or 1
                pct = int(100 * min(1.0, v[1] / f) + 1e-6)   # 切り捨て（しきい値の判定を従来と同じにする）
                if pct >= MIN_PCT:
                    nd["p"].setdefault(pid, []).extend([di, round(v[0] * 1000), pct])
            added += 1
        sort_dates(nd)
        # 途中で落ちても進捗を失わないよう都度保存
        write_json(os.path.join(cdir, fname), nd)
    if fname == "ndvi_next.json":              # 最新まで追いついたので入れ替える
        write_json(os.path.join(cdir, "ndvi.json"), nd)
        if os.path.exists(os.path.join(cdir, "ndvi_next.json")):
            os.remove(os.path.join(cdir, "ndvi_next.json"))
        log(f"{cid}: 新しい雲判定で計算し直し完了 ({len(nd['dates'])}日, 区画{len(pids)})")
    elif added:
        log(f"{cid}: +{added}日 (区画{len(pids)})")
    return cid, added


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-minutes", type=float, default=300)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--backend", choices=["gee", "fake"], default="gee")
    ap.add_argument("--fake-csv")
    ap.add_argument("--cells", help="カンマ区切りでセルIDを限定（テスト用）")
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
    index["updated"] = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    index["source"] = "Sentinel-2 L2A (Copernicus) / Google Earth Engine"
    write_json(os.path.join(data_dir, "index.json"), index)
    log(f"完了: {done}セル処理, 観測日を合計 {total} 追加")

if __name__ == "__main__":
    main()
