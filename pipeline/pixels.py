"""圃場内マップ用の画素データ（10m画素ごとの NDVI）を作る。

各セル・各年について、期間（config の pixel_window, 既定 5/15〜9/25）の Sentinel-2 の晴れた日を
1枚の PNG（グレースケール）に縦に積んで cells/<id>/px_<年>.png に保存する。
  画素値 0 = 雲・田以外、1〜255 = NDVI（-0.2〜1.0 を 1〜255 に割り当て）
座標は Web メルカトル（EPSG:3857）で、地図にそのまま重ねられる。
説明は cells/<id>/px_<年>.json:
  {"v":1, "mask":"<雲判定の方式>", "dates":[PNGに入っている日], "src_dates":[確認済みの日],
   "w":幅, "h":1日分の高さ, "x0":左端, "y1":上端, "px":画素の大きさ(3857のm), "done":期間が終わって確定したか}
対象は今年と昨年（今年の期間がまだ始まっていなければ昨年と一昨年）。それより古い年のファイルは消す。
"""
import os, io, math, datetime as dt
from common import write_json, read_json, with_retry

R = 6378137.0
MIN_CLEAR = 0.2          # セル内の田の画素のうち晴れている割合がこれ未満の日は保存しない
BANDS_PER_CALL = 8       # 1回の computePixels で取る日数


def merc(lon, lat):
    return R * math.radians(lon), R * math.log(math.tan(math.pi / 4 + math.radians(lat) / 2))


def grid_for(bbox):
    """セルの範囲を覆う 3857 の格子（地上で約10m）"""
    w, s, e, n = bbox
    px = 10.0 / math.cos(math.radians((s + n) / 2))
    x0, y0 = merc(w, s); x1, y1 = merc(e, n)
    x0 = math.floor(x0 / px) * px; y1 = math.ceil(y1 / px) * px
    return {"x0": x0, "y1": y1, "px": px, "w": int(math.ceil((x1 - x0) / px)), "h": int(math.ceil((y1 - y0) / px))}


def windows(cfg, today):
    """[(年, 期間の始め, 期間の終わり), ...]"""
    (sm, sd), (em, ed) = [map(int, x.split("-")) for x in cfg.get("pixel_window", ["05-15", "09-25"])]
    y = today.year if today >= dt.date(today.year, sm, sd) else today.year - 1
    return [(yy, dt.date(yy, sm, sd), dt.date(yy, em, ed)) for yy in (y - 1, y)]


class GEEPixels:
    """GEE から画素を取る（backend は update_ndvi.GEEBackend）"""
    def __init__(self, backend):
        self.b = backend; self.ee = backend.ee

    def _req(self, img, g):
        return {"expression": img, "fileFormat": "NUMPY_NDARRAY",
                "grid": {"dimensions": {"width": g["w"], "height": g["h"]},
                         "affineTransform": {"scaleX": g["px"], "shearX": 0, "translateX": g["x0"],
                                             "shearY": 0, "scaleY": -g["px"], "translateY": g["y1"]},
                         "crsCode": "EPSG:3857"}}

    def paddy(self, fc, g):
        """田の画素（区画の形を塗ったもの）: 0/1 の配列"""
        import numpy as np
        ee = self.ee
        img = ee.Image(0).byte().paint(ee.FeatureCollection(fc), 1).rename("m")
        return np.asarray(with_retry(lambda: ee.data.computePixels(self._req(img, g)))["m"], dtype=np.uint8)

    def ndvi(self, bbox, dates, g):
        """{date: uint8 配列（0=雲, 1〜255=NDVI）}"""
        import numpy as np
        ee = self.ee
        col = self.b._col_s2(bbox, dates[0], (dt.date.fromisoformat(dates[-1]) + dt.timedelta(days=1)).isoformat())
        out = {}
        for i in range(0, len(dates), BANDS_PER_CALL):
            chunk = dates[i:i + BANDS_PER_CALL]
            imgs = [col.filterDate(d, ee.Date(d).advance(1, "day")).mosaic().select("NDVI")
                    .clamp(-0.2, 1.0).add(0.2).divide(1.2).multiply(254).add(1).round().unmask(0).byte()
                    .rename("d" + d.replace("-", "")) for d in chunk]
            arr = with_retry(lambda: ee.data.computePixels(self._req(ee.Image.cat(imgs), g)))
            for d in chunk:
                out[d] = np.asarray(arr["d" + d.replace("-", "")], dtype=np.uint8)
        return out


def read_layers(png_p, meta):
    import numpy as np
    from PIL import Image
    if not meta.get("dates") or not os.path.exists(png_p):
        return {}
    a = np.asarray(Image.open(png_p).convert("L"))
    h = meta["h"]
    return {d: a[i * h:(i + 1) * h] for i, d in enumerate(meta["dates"])}


def write_png(png_p, arrays):
    import numpy as np
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(np.vstack(arrays), mode="L").save(buf, format="PNG", optimize=True)
    with open(png_p, "wb") as f:
        f.write(buf.getvalue())


def update_cell(src, cfg, data_dir, cell, today, mask, log):
    """1セルの画素データを、まだ取っていない日だけ足す。取った日数を返す。"""
    cdir = os.path.join(data_dir, "cells", cell["id"])
    nd = read_json(os.path.join(cdir, "ndvi.json"))
    parcels = read_json(os.path.join(cdir, "parcels.geojson"))
    if not nd or not parcels or not parcels.get("features"):
        return 0
    fc = {"type": "FeatureCollection", "features": [{"type": "Feature", "properties": {}, "geometry": f["geometry"]} for f in parcels["features"]]}
    g = grid_for(cell["bbox"])
    added = 0; keep = set(); paddy = None
    for year, ws, we in windows(cfg, today):
        keep.add(year)
        meta_p = os.path.join(cdir, f"px_{year}.json"); png_p = os.path.join(cdir, f"px_{year}.png")
        dates = [d for d in nd.get("dates", []) if ws.isoformat() <= d <= we.isoformat()]
        meta = read_json(meta_p) or {}
        if meta.get("mask") != mask or any(meta.get(k) != g[k] for k in g):
            meta = {}                                  # 方式か格子が変わったら作り直し
        if meta.get("done"):
            continue
        new = [d for d in dates if d not in set(meta.get("src_dates", []))]
        done = today > we + dt.timedelta(days=20)
        if not new and meta:
            if done:
                meta["done"] = True; write_json(meta_p, meta)
            continue
        layers = read_layers(png_p, meta) if meta else {}
        if new:
            if paddy is None:
                paddy = src.paddy(fc, g)
            n_paddy = max(1, int(paddy.sum()))
            for d, a in src.ndvi(cell["bbox"], new, g).items():
                a = a * paddy
                if (a > 0).sum() / n_paddy >= MIN_CLEAR:
                    layers[d] = a
        kept = sorted(layers)
        if kept:
            write_png(png_p, [layers[d] for d in kept])
        elif os.path.exists(png_p):
            os.remove(png_p)
        write_json(meta_p, {"v": 1, "mask": mask, "dates": kept, "src_dates": sorted(set(meta.get("src_dates", [])) | set(new)),
                            **g, "done": done and set(dates) <= set(meta.get("src_dates", [])) | set(new)})
        added += len(new)
        log(f"{cell['id']}: 圃場内マップ {year} +{len(new)}日（保存 {len(kept)}日）")
    for f in os.listdir(cdir):                        # 対象外になった年のファイルは消す
        if f.startswith("px_") and f[3:7].isdigit() and int(f[3:7]) not in keep:
            os.remove(os.path.join(cdir, f))
    return added
