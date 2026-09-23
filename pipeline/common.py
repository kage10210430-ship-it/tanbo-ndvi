import json, math, os
import yaml, requests
import pandas as pd
import geopandas as gpd

def load_config(path="config.yaml"):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)

def cell_id(lon, lat, cfg):
    cx = math.floor(lon / cfg["cell_deg_lon"]); cy = math.floor(lat / cfg["cell_deg_lat"])
    return f"{cx}_{cy}"

def fetch_source(src, cache_dir=".cache"):
    """URL ならダウンロードしてローカルパスを返す。"""
    if src.startswith("http://") or src.startswith("https://"):
        os.makedirs(cache_dir, exist_ok=True)
        name = os.path.join(cache_dir, os.path.basename(src.split("?")[0]) or "src.bin")
        if not os.path.exists(name):
            with requests.get(src, stream=True, timeout=600) as r:
                r.raise_for_status()
                with open(name, "wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
        return name
    return src

def read_parcels(cfg):
    frames = []
    for src in cfg["parcel_sources"]:
        path = fetch_source(src)
        kw = {}
        if cfg.get("bbox"):
            kw["bbox"] = tuple(cfg["bbox"])
        frames.append(gpd.read_file(path, engine="pyogrio", **kw))
    gdf = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs=frames[0].crs)
    if gdf.crs is None or gdf.crs.to_epsg() != 4326:
        gdf = gdf.set_crs(4326) if gdf.crs is None else gdf.to_crs(4326)
    lt = cfg.get("land_type_field")
    if cfg.get("only_paddy") and lt in gdf.columns:
        gdf = gdf[gdf[lt].astype(str).str.split(".").str[0] == "100"]
    idf = cfg.get("id_field")
    gdf["pid"] = gdf[idf].astype(str) if idf in gdf.columns else [f"p{i}" for i in range(len(gdf))]
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty]
    cols = ["pid", "geometry"] + ([lt] if lt in gdf.columns else [])
    return gdf[cols].reset_index(drop=True)

def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, separators=(",", ":"))

def read_json(path, default=None):
    if not os.path.exists(path):
        return default
    with open(path, encoding="utf-8") as f:
        return json.load(f)
