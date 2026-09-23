import json, math, os, re, glob, zipfile, shutil, time, random
import yaml, requests
import pandas as pd
import geopandas as gpd

CACHE = ".cache"
ID_CANDIDATES = ["polygon_uuid", "polygon_id", "fude_id", "id", "ID"]
LT_CANDIDATES = ["land_type", "LAND_TYPE", "地目", "耕地種類", "land_type_cd"]

def load_config(path="config.yaml"):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)

def cell_id(lon, lat, cfg):
    cx = math.floor(lon / cfg["cell_deg_lon"]); cy = math.floor(lat / cfg["cell_deg_lat"])
    return f"{cx}_{cy}"

def _download(url):
    os.makedirs(CACHE, exist_ok=True)
    name = os.path.join(CACHE, os.path.basename(url.split("?")[0]) or "src.bin")
    if not os.path.exists(name):
        print("ダウンロード:", url, flush=True)
        with requests.get(url, stream=True, timeout=900, headers={"User-Agent": "Mozilla/5.0"}) as r:
            r.raise_for_status()
            with open(name + ".part", "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
        os.replace(name + ".part", name)
    return name

def _materialize(sources):
    """URL/パスの一覧を、読めるファイル（.fgb/.geojson/.shp）の一覧にする。
    分割zip（xxx.zip.001, .002 …）は連結してから展開する。"""
    local = [(_download(s) if re.match(r"https?://", s) else s) for s in sources]
    groups, singles = {}, []
    for p in local:
        m = re.match(r"(.+\.zip)\.\d{3}$", p)
        if m: groups.setdefault(m.group(1), []).append(p)
        else: singles.append(p)
    archives = []
    for base, parts in groups.items():
        if not os.path.exists(base):
            with open(base, "wb") as out:
                for part in sorted(parts):
                    with open(part, "rb") as f: shutil.copyfileobj(f, out)
        archives.append(base)
    files = []
    for p in singles + archives:
        if p.lower().endswith(".zip"):
            dest = p[:-4] + "_unzip"
            if not os.path.isdir(dest):
                with zipfile.ZipFile(p) as z: z.extractall(dest)
            for ext in ("fgb", "geojson", "json", "shp"):
                files += glob.glob(os.path.join(dest, "**", f"*.{ext}"), recursive=True)
        else:
            files.append(p)
    return files

def _pick(cols, cands):
    for c in cands:
        if c in cols: return c
    return None

def read_parcels(cfg):
    files = _materialize(cfg["parcel_sources"])
    if not files:
        raise SystemExit("区画データが見つかりません。config.yaml の parcel_sources を確認してください。")
    frames = []
    for path in files:
        kw = {"bbox": tuple(cfg["bbox"])} if cfg.get("bbox") else {}
        g = gpd.read_file(path, engine="pyogrio", **kw)
        if len(g):
            g = g.set_crs(4326) if g.crs is None else g.to_crs(4326)
            frames.append(g)
        print(f"  {os.path.basename(path)}: {len(g)} 区画", flush=True)
    if not frames:
        raise SystemExit("指定範囲（bbox）に区画がありません。bbox の数字（西, 南, 東, 北）を確認してください。")
    gdf = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs=4326)
    print("列:", list(gdf.columns), flush=True)

    lt = cfg.get("land_type_field") if cfg.get("land_type_field") in gdf.columns else _pick(gdf.columns, LT_CANDIDATES)
    if cfg.get("only_paddy"):
        if lt:
            v = gdf[lt].astype(str).str.strip()
            gdf = gdf[(v.str.split(".").str[0] == "100") | (v == "田")]
            print(f"田だけに絞り込み（列 {lt}）: {len(gdf)} 区画", flush=True)
        else:
            print("※ 田/畑を示す列が見つからないため、全区画を対象にします", flush=True)
    idf = cfg.get("id_field") if cfg.get("id_field") in gdf.columns else _pick(gdf.columns, ID_CANDIDATES)
    gdf["pid"] = gdf[idf].astype(str) if idf else [f"p{i}" for i in range(len(gdf))]
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty]
    gdf = gdf.drop_duplicates("pid")
    cols = ["pid", "geometry"] + ([lt] if lt else [])
    out = gdf[cols].reset_index(drop=True)
    if lt and lt != "lt_src":
        out = out.rename(columns={lt: "lt_src"})
    return out

def round_coords(coords, nd=5):
    """GeoJSON の座標を小数 nd 桁（5桁 ≒ 1m）に丸め、丸めて重なった連続点は1つにする"""
    if isinstance(coords[0], (int, float)):
        return [round(coords[0], nd), round(coords[1], nd)]
    out = [round_coords(c, nd) for c in coords]
    if out and isinstance(out[0][0], (int, float)):
        out = [out[0]] + [p for a, p in zip(out, out[1:]) if p != a]
    return out

def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, separators=(",", ":"))

def read_json(path, default=None):
    if not os.path.exists(path):
        return default
    with open(path, encoding="utf-8") as f:
        return json.load(f)


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
