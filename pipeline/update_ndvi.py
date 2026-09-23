"""各セルの区画について、まだ計算していない観測日の Sentinel-2 NDVI（区画平均）を
Google Earth Engine で計算し、cells/<id>/ndvi.json に追記する。

  python pipeline/update_ndvi.py --max-minutes 300 --workers 4

ndvi.json の形式:
  {"dates": ["2025-09-25", ...],
   "p": {"<pid>": [[ndvi, count] | null, ...]}}     # dates と同じ並び
有効画素率は表示側で count / full（parcels.geojson の full）から出す。

--backend fake --fake-csv <file> を付けると GEE を使わずに CSV（pid,date,mean,count）から
同じ形式を作る（動作確認用）。
"""
import argparse, os, sys, json, time, datetime as dt
from concurrent.futures import ThreadPoolExecutor, as_completed
from common import load_config, write_json, read_json

MAX_FEATURES_PER_CALL = 4500      # getInfo の要素数上限(5000)に余裕を持たせる


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
        ee = self.ee
        region = ee.Geometry.Rectangle(bbox)
        def prep(img):
            scl = img.select("SCL")
            good = scl.eq(2).Or(scl.eq(4)).Or(scl.eq(5)).Or(scl.eq(6)).Or(scl.eq(7))
            return (img.normalizedDifference(["B8", "B4"]).rename("NDVI").updateMask(good)
                    .copyProperties(img, ["system:time_start"]))
        return (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                .filterBounds(region).filterDate(start, end)
                .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", self.cfg["max_scene_cloud"]))
                .map(prep))

    def list_dates(self, bbox, start, end):
        """期間内の観測日（UTC日付, YYYY-MM-DD）を返す"""
        ts = self._col(bbox, start, end).aggregate_array("system:time_start").getInfo()
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
        res = table.select(["pid", "date", "mean", "count"], None, False).getInfo()
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
def process_cell(backend, cfg, data_dir, cell, start_default, end, log):
    cid = cell["id"]; cdir = os.path.join(data_dir, "cells", cid)
    nd = read_json(os.path.join(cdir, "ndvi.json"), {"dates": [], "p": {}})
    inner = read_json(os.path.join(cdir, "inner.geojson"))
    if not inner:
        return cid, 0
    pids = [f["properties"]["pid"] for f in inner["features"]]
    for pid in pids:                       # 新しい区画が増えていたら列を揃える
        if pid not in nd["p"]:
            nd["p"][pid] = [None] * len(nd["dates"])
    have = set(nd["dates"])
    start = (dt.date.fromisoformat(nd["dates"][-1]) + dt.timedelta(days=1)).isoformat() if nd["dates"] else start_default
    dates = [d for d in backend.list_dates(cell["bbox"], start, end) if d not in have]
    if not dates:
        return cid, 0
    per_call = max(1, MAX_FEATURES_PER_CALL // max(1, len(pids)))
    added = 0
    for i in range(0, len(dates), per_call):
        chunk = dates[i:i + per_call]
        res = backend.stats(cell["bbox"], inner, chunk)
        for d in chunk:
            nd["dates"].append(d)
            for pid in pids:
                v = res.get(d, {}).get(pid)
                nd["p"][pid].append(v if v else None)
            added += 1
        # 途中で落ちても進捗を失わないよう都度保存
        write_json(os.path.join(cdir, "ndvi.json"), nd)
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
    def last_date(c):
        nd = read_json(os.path.join(data_dir, "cells", c["id"], "ndvi.json"))
        return nd["dates"][-1] if nd and nd["dates"] else ""
    cells = sorted(cells, key=last_date)

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
