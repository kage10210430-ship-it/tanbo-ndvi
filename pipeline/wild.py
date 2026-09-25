"""獣害の起きやすさの手がかり（田ごと）を作る。生育傾向マップの圃場の評価で、山際の減点に使う。

cells/<id>/wild.json: {"v":1, "src":"worldcover-v200", "p":{"<pid>":[f30, f100, f200, dirs, paddy]}}
  f30 / f100 / f200 = 田の縁から 0〜30m / 30〜100m / 100〜200m の輪の中の森の割合（%）
  dirs  = 田の中心から見た16方位（北から時計回り、ビット0=北）のうち、200m以内の輪で森が25%以上ある方位（ビット）
  paddy = 田の縁から300mの輪の中の、ほかの田（筆ポリゴン）の割合（%）。少ないほど周りに田がなく孤立している
森は ESA WorldCover 2021（10m, 樹林 = 10）。区画は隣のセルの分も含めて塗る。形は変わらないので1回だけ作る。
"""
import os, math
from common import write_json, read_json, with_retry
from pixels import grid_for

VERSION = 1
PAD_DEG = (0.0045, 0.0036)       # セルの外に広げる幅（経度, 緯度）: 約400m
RINGS_M = (30, 100, 200)
PADDY_M = 300
DIR_MIN = 0.25


class GEEWild:
    def __init__(self, backend, px):
        self.ee = backend.ee; self.px = px

    def forest(self, g):
        """格子 g の森（1）・それ以外（0）"""
        import numpy as np
        ee = self.ee
        img = ee.ImageCollection("ESA/WorldCover/v200").first().select("Map").eq(10).unmask(0).byte().rename("f")
        return np.asarray(with_retry(lambda: ee.data.computePixels(self.px._req(img, g)))["f"], dtype=np.uint8)


class FakeWild:
    """動作確認用: セルの西の端から200mは森"""
    def forest(self, g):
        import numpy as np
        a = np.zeros((g["h"], g["w"]), np.uint8); a[:, :int(200 / 10) + int(PAD_DEG[0] * 111000 * 0.81 / 10)] = 1
        return a


def neighbors(index, cid):
    x, y = map(int, cid.split("_"))
    ids = {c["id"] for c in index["cells"]}
    return [f"{x + dx}_{y + dy}" for dx in (-1, 0, 1) for dy in (-1, 0, 1) if f"{x + dx}_{y + dy}" in ids]


def update_cell(src, cfg, data_dir, index, cell, log, deadline=None):
    """1セルの wild.json を作る。作ったら 1 を返す"""
    import numpy as np, shapely
    from shapely.geometry import shape
    cdir = os.path.join(data_dir, "cells", cell["id"])
    parcels = read_json(os.path.join(cdir, "parcels.geojson"))
    if not parcels or not parcels.get("features"):
        return 0
    w, s, e, n = cell["bbox"]
    g = grid_for([w - PAD_DEG[0], s - PAD_DEG[1], e + PAD_DEG[0], n + PAD_DEG[1]])   # 格子1つ ≒ 地上10m
    def to_grid(xy):
        x = 6378137.0 * np.radians(xy[:, 0]); y = 6378137.0 * np.log(np.tan(np.pi / 4 + np.radians(xy[:, 1]) / 2))
        return np.column_stack([(x - g["x0"]) / g["px"], (g["y1"] - y) / g["px"]])
    forest = src.forest(g).astype(bool)
    # 田（まわりのセルの区画も）を塗る
    paddy = np.zeros((g["h"], g["w"]), bool)
    yy, xx = np.mgrid[0:g["h"], 0:g["w"]]; cx = xx + 0.5; cy = yy + 0.5
    for cid in neighbors(index, cell["id"]):
        pp = read_json(os.path.join(data_dir, "cells", cid, "parcels.geojson")) or {"features": []}
        for f in pp["features"]:
            try:
                gg = shapely.transform(shape(f["geometry"]), to_grid).buffer(0)
                x0, y0, x1, y1 = gg.bounds
                c0, c1, r0, r1 = max(0, math.floor(x0)), min(g["w"], math.ceil(x1)), max(0, math.floor(y0)), min(g["h"], math.ceil(y1))
                if c1 > c0 and r1 > r0:
                    paddy[r0:r1, c0:c1] |= shapely.contains_xy(gg, cx[r0:r1, c0:c1], cy[r0:r1, c0:c1])
            except Exception:
                continue
    out = {}
    for f in parcels["features"]:
        pid = f["properties"]["pid"]
        try:
            gg = shapely.transform(shape(f["geometry"]), to_grid).buffer(0)
            far = gg.buffer(PADDY_M / 10.0)
            x0, y0, x1, y1 = far.bounds
            c0, c1, r0, r1 = max(0, math.floor(x0)), min(g["w"], math.ceil(x1)), max(0, math.floor(y0)), min(g["h"], math.ceil(y1))
            X, Y = cx[r0:r1, c0:c1], cy[r0:r1, c0:c1]; F = forest[r0:r1, c0:c1]; P = paddy[r0:r1, c0:c1]
            inside = shapely.contains_xy(gg, X, Y)
            dist = np.asarray(shapely.distance(gg, shapely.points(X.ravel(), Y.ravel()))).reshape(X.shape) * 10.0   # 田の縁からの距離（m）
            dist[inside] = 0
            row = []
            lo = 0
            for hi in RINGS_M:
                ring = (dist > lo) & (dist <= hi) & ~inside
                row.append(round(100 * F[ring].mean()) if ring.any() else 0); lo = hi
            # 16方位（田の中心から見た向き）で、200m以内の輪に森が DIR_MIN 以上ある方位
            ccx, ccy = gg.centroid.x, gg.centroid.y
            ang = (np.degrees(np.arctan2(X - ccx, -(Y - ccy))) + 360 + 11.25) % 360 // 22.5
            near = (dist > 0) & (dist <= RINGS_M[-1]) & ~inside
            mask = 0
            for k in range(16):
                sec = near & (ang == k)
                if sec.sum() >= 5 and F[sec].mean() >= DIR_MIN:
                    mask |= 1 << k
            row.append(mask)
            ring = (dist > 0) & (dist <= PADDY_M) & ~inside
            others = P & ~inside
            row.append(round(100 * others[ring].mean()) if ring.any() else 0)
            out[pid] = row
        except Exception:
            continue
    write_json(os.path.join(cdir, "wild.json"), {"v": VERSION, "src": "worldcover-v200", "p": out})
    return 1


def need(data_dir, cell):
    m = read_json(os.path.join(data_dir, "cells", cell["id"], "wild.json")) or {}
    return m.get("v") != VERSION
