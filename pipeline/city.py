"""田ごとの市町を決め、市町ごとの獣害の被害の多さと合わせて data/city.json にする。圃場の評価の獣害の減点に使う。

市町の境界は 国土数値情報（行政区域 N03, 国土交通省）を GeoJSON にしたもの（config.yaml の city_sources）。
田の代表点（田の中の1点）がどの市町に入るかで決める。境界は変わらないので1回だけ作る。
被害の数字は pipeline/wild_city.json（市町の資料から手で写したもの・出典つき）をそのまま入れる。

data/city.json: {"v":1, "names":{"18201":"福井市",…}, "dmg":{…wild_city.json…},
                 "cells":{"<cell>":["18201", {"18202":["pidの頭8文字",…]}]}}
  セルの中の田はふつう1つ目の市町。ほかの市町に入る田だけを2つ目に書く（pid の頭8文字。セルの中で重なるときは pid 全体）
"""
import os, json
from common import write_json, read_json, _download

VERSION = 1
HERE = os.path.dirname(os.path.abspath(__file__))


def load_bounds(cfg):
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
    from shapely.prepared import prep
    out = []
    for src in cfg.get("city_sources") or []:
        path = _download(src) if src.startswith("http") else src
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        for ft in d["features"]:
            # この GeoJSON は、飛び地や島の輪がすべて1つの多角形の「穴」として入っているので、輪ごとの多角形にしてから合わせる
            gm = ft["geometry"]; polys = gm["coordinates"] if gm["type"] == "MultiPolygon" else [gm["coordinates"]]
            g = unary_union([Polygon(r).buffer(0) for poly in polys for r in poly if len(r) >= 4])
            p = ft["properties"]
            out.append((p["N03_007"], p["N03_004"], g, prep(g)))
    return out


def build(cfg, data_dir, log=print):
    from shapely.geometry import shape
    bounds = load_bounds(cfg)
    if not bounds:
        log("city: city_sources がないので作りません"); return 0
    index = read_json(os.path.join(data_dir, "index.json")) or {"cells": []}
    cells, miss = {}, 0
    for c in index["cells"]:
        pp = read_json(os.path.join(data_dir, "cells", c["id"], "parcels.geojson")) or {"features": []}
        got = {}
        for f in pp["features"]:
            pid = f["properties"]["pid"]
            try:
                pt = shape(f["geometry"]).representative_point()
            except Exception:
                continue
            code = next((k for k, _, _, pg in bounds if pg.contains(pt)), None)
            if code is None:      # 海岸・境界の線の上など: 一番近い市町
                code = min(bounds, key=lambda b: b[2].distance(pt))[0]; miss += 1
            got.setdefault(code, []).append(pid)
        if not got:
            continue
        main = max(got, key=lambda k: len(got[k]))
        short = {}
        heads = [p[:8] for ps in got.values() for p in ps]
        for k, ps in got.items():
            if k == main:
                continue
            short[k] = [p[:8] if heads.count(p[:8]) == 1 else p for p in ps]
        cells[c["id"]] = [main, short] if short else [main]
    dmg = read_json(os.path.join(HERE, "wild_city.json")) or {}
    write_json(os.path.join(data_dir, "city.json"), {"v": VERSION, "src": "国土数値情報 行政区域（N03）国土交通省",
                                                      "names": {k: n for k, n, _, _ in bounds}, "dmg": dmg, "cells": cells})
    log(f"city: {len(cells)} セル（境界の外で一番近い市町にした田 {miss}）")
    return 1


def need(data_dir):
    m = read_json(os.path.join(data_dir, "city.json")) or {}
    dmg = read_json(os.path.join(HERE, "wild_city.json")) or {}
    return m.get("v") != VERSION or m.get("dmg") != dmg


if __name__ == "__main__":
    from common import load_config
    cfg = load_config()
    build(cfg, os.path.join(cfg["site_dir"], "data"))
