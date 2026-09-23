"""区画データ → 配信用グリッド（cells/<id>/parcels.geojson, inner.geojson）と index.json。
区画データが更新されたとき、または初回に実行する。既存の ndvi.json は残す。"""
import os, json
import geopandas as gpd
from shapely.geometry import mapping
from common import load_config, cell_id, read_parcels, write_json, read_json, round_coords

def round_coords_geom(geom):
    g = json.loads(json.dumps(geom))
    g["coordinates"] = round_coords(g["coordinates"])
    return g

def main():
    cfg = load_config()
    data_dir = os.path.join(cfg["site_dir"], "data")
    gdf = read_parcels(cfg)
    print(f"区画 {len(gdf)} 件")
    utm = gdf.estimate_utm_crs()
    proj = gdf.to_crs(utm)
    inner = proj.geometry.buffer(-cfg["edge_m"])
    gdf["inner_area"] = inner.area.values
    keep = gdf["inner_area"] > 50                      # 縮めて消える細い区画は除く
    gdf = gdf[keep].copy()
    gdf["inner_wgs"] = gpd.GeoSeries(inner[keep].values, crs=utm).to_crs(4326).values
    cen = gdf.geometry.representative_point()
    gdf["cell"] = [cell_id(p.x, p.y, cfg) for p in cen]
    lt = cfg.get("land_type_field")
    cells = []
    for cid, g in gdf.groupby("cell"):
        feats, inner_feats = [], []
        for _, r in g.iterrows():
            props = {"pid": r["pid"], "full": round(r["inner_area"] / 100.0, 1)}   # 10m画素の期待数
            if lt in g.columns and r[lt] is not None:
                try: props["lt"] = int(float(r[lt]))
                except (TypeError, ValueError): pass
            feats.append({"type": "Feature", "properties": props,
                          "geometry": round_coords_geom(mapping(r.geometry.simplify(0.000005, preserve_topology=True)))})
            inner_feats.append({"type": "Feature", "properties": {"pid": r["pid"]},
                                "geometry": json.loads(json.dumps(mapping(r["inner_wgs"].simplify(0.00001, preserve_topology=True))))})
        write_json(os.path.join(data_dir, "cells", cid, "parcels.geojson"), {"type": "FeatureCollection", "precision": 5, "features": feats})
        write_json(os.path.join(data_dir, "cells", cid, "inner.geojson"), {"type": "FeatureCollection", "features": inner_feats})
        b = g.total_bounds
        cells.append({"id": cid, "bbox": [round(float(x), 6) for x in b], "n": int(len(g))})
    index = read_json(os.path.join(data_dir, "index.json"), {}) or {}
    index.update({"cells": sorted(cells, key=lambda c: c["id"]), "n_parcels": int(len(gdf)),
                  "cell_deg_lon": cfg["cell_deg_lon"], "cell_deg_lat": cfg["cell_deg_lat"],
                  "bbox": [round(float(x), 6) for x in gdf.total_bounds]})
    write_json(os.path.join(data_dir, "index.json"), index)
    print(f"グリッド {len(cells)} セル → {data_dir}")

if __name__ == "__main__":
    main()
