"""機械（田植機・コンバイン）のハマりやすさの手がかり（田ごと・10m画素ごと）を作る。生育傾向マップの圃場の評価と「ハマりやすさマップ」に使う。

3つの部分を別々に作り、できた部分から保存する（時間切れ・取得の失敗があっても、次回に足りない部分だけ作り直す）:
  土   : 農研機構 日本土壌インベントリーの土壌図タイル（z15 の色 → 土壌の記号。色の表は soil_rgb.json）。
         田の中の面積の割合で、土の軟らかさ・水はけの悪さ（0〜1）を平均する（泥炭土 1.0・グライ低地土 0.75・灰色低地土 0.35・褐色低地土 0.15 など）
  地形 : 国土地理院の標高タイル（dem5a → 5b → 5c → dem_png(10m) の順に埋める）。取れなければ GEE の Copernicus DEM（30m）。
         周りの田との高さの差・周りの中の低さの順位・山際（200m以内の高い所）・地形の湿りやすさ（TWI）・田の中の低い所（5m レーザーのときだけ）
  乾き : 衛星（GEE）。田に作物のない時期（春 3/1〜4/25・秋 10/10〜12/10、2022年から）の晴れた日・レーダーの日ごとに、
         同じ日の周り 6km 四方の畑・田（ESA WorldCover の耕地）の中央値と比べて、どれだけ暗い（SWIR, Sentinel-2 B12）・
         レーダーの反射が強い（Sentinel-1 VV）・水が溜まっている かを数える（雨の日の違い・軌道の違いを打ち消す）。
         雪の日（画素の雪・SCL の雪、ERA5-Land の積雪）・凍った日（レーダー）・周りが一面水の日は使わない。
         観測の時刻から前48時間の雨（ERA5-Land 1時間ごと）が 1mm 未満の日だけの値も別に持つ
         → 湿りやすさの指数 wi（標準偏差の単位。0 = 周りのふつう、+ ほど乾きにくい）

cells/<id>/wet.png : 格子は trend.png と同じ（grid_for(cell.bbox), EPSG:3857, 約10m）。layers の順に縦に積んだグレースケール（0 = なし）
    wi : 湿りやすさ指数 1〜255 = −3〜+3
    dz : 田の中の高さの差（傾きを除いた、その田の中央値との差）128 + dz/2cm（±2.5m）。5m レーザー（dem5a）の田だけ
cells/<id>/wet.json:
  {"v":1, "w","h","x0","y1","px", "layers":[...],
   "soil":{"src","tried","names":{記号: 名前}, "match":[色が一致した%, 近い色%], "p":{pid:[軟らかさ×100|null, 主な記号, 不明%, 主な記号の割合%]} | null},
   "terr":{"src","tried","cover":[5a%,5b/5c%,10m%], "p":{pid:[標高cm, 周りの田との差cm, 周りの中の低さ%, 低い所%, 低い所の向き(8方位 北=0, なし −1),
                                                     田の中の凹凸cm, 山際%, TWI×10, 元(0=5aレーザー,1=5b/5c,2=10m,3=GEE 30m)]} | null},
   "wet":{"src","seasons":[済んだ時期], "fail":{時期: [失敗回数, 最後に失敗した日]}, "p":{pid:[wi中央値×100, wi上位1割×100, wi>1 の割合%, 乾いた日の SWIR の差×1000|null,
                                                                  レーダーの差 dB×10|null, 水が見えた割合%|null, 使った観測の数]}}}
<wet_state_dir>/<id>.npz（公開しない。Actions では wet-state ブランチ）: 乾きの時期ごとの合計を足したもの（int32, 田の外は0。キー s2_A, s1_N … と seasons）。
  新しい時期が終わったら、その時期だけ GEE で取って足す（古い時期は取り直さない。wet_seasons は最初に取る時期の数）
  土・地形は1回だけ（取れなかったら RETRY_DAYS 日あとにもう一度）。乾きで2回続けて失敗した時期も RETRY_DAYS 日あとにもう一度。
  trend.py からは2回に分けて呼ぶ: 先に全セルの土・地形（タイルだけで軽い）、残りの時間で乾き（GEE で重い）。
--backend fake（trend.py）では FakeTiles・FakeSat（作り物のデータ）で作る。GEE の確かめは  python pipeline/wet.py --probe --cells <id>
"""
import os, io, math, time, json, threading, datetime as dt
from common import write_json, read_json, with_retry
from pixels import grid_for, read_layers, write_png

VERSION = 1
PAD_DEG = (0.0056, 0.0045)        # セルの外に広げる幅（経度, 緯度）: 約500m（周りの田と比べるため）
RETRY_DAYS = 28                   # 土・地形のタイルが取れなかったセルを、もう一度試すまでの日数
TILE = 256; RM = 6378137.0; ORIG = math.pi * RM
UA = "tanbo-ndvi (+https://github.com/; paddy field NDVI map, Fukui)"
URLS = {   # 上から順に試す（同じ場所の予備）
    "soil": ["https://soil-inventory.rad.naro.go.jp/tile/figure/{z}/{x}/{y}.png",
             "https://soil-inventory.dc.affrc.go.jp/tile/figure/{z}/{x}/{y}.png"],
    "dem5a": ["https://cyberjapandata.gsi.go.jp/xyz/dem5a_png/{z}/{x}/{y}.png"],
    "dem5b": ["https://cyberjapandata.gsi.go.jp/xyz/dem5b_png/{z}/{x}/{y}.png"],
    "dem5c": ["https://cyberjapandata.gsi.go.jp/xyz/dem5c_png/{z}/{x}/{y}.png"],
    "dem_png": ["https://cyberjapandata.gsi.go.jp/xyz/dem_png/{z}/{x}/{y}.png"],
}
MIN_GAP = 0.2                     # タイルを取る間隔（秒）。全部のスレッドで合わせて 5件/秒 以下
DOWN_AFTER = 8                    # 同じサーバーで続けてこの回数失敗したら、今回の実行ではもう使わない
EDGE_M = 5.0                      # 田の中の低い所は、田の縁から 5m を除いて見る（畦畔・溝）
LOW_M = 0.10                      # 田の中央値よりこれ以上低い所を「低い所」にする（傾きを除いたあと）
LOW_MIN_PX = 3                    # 低い所は 3画素（z15 で約45m²）以上つながったものだけ
IN_COVER = 0.7                    # 田の中の低い所は、dem5a（レーザー）が田のこの割合以上を覆うときだけ
REL_M, PCT_M, HILL_M, HILL_DZ = 300, 500, 200, 3.0
# 乾き（衛星）
SEAS = (("pre", "03-01", "04-25"), ("post", "10-10", "12-10"))
WATER_LAST = "04-16"              # 春は 4/15 までの水だけ数える（そのあとは代かき前の水入れ）
NDVI_BARE = 0.35; C2 = 0.6; C1 = 6.0
REF_DEG = (0.033, 0.027)          # 周りの比べる範囲（セルの中心から ±経度, ±緯度）: 約 6km 四方
REF_SCALE = 40; MIN_REF = 150; THR_CLR = 0.15; THR_SNOW = 0.03; THR_WF = 0.25; DRY_MM = 1.0; FROZEN_K = 273.65; MAX_ZEN = 65
SNOW_M = {"s2": 0.02, "s1": 0.005}  # ERA5 の積雪（m, セルの中心の格子）。S2 は画素ごとの雪も見るのでゆるく、雪の見えないレーダーはきびしく
RAIN_H = 48                       # 観測の時刻から前 48 時間の雨（ERA5-Land 1時間ごと）
BARE_PCT = {"pre": 50, "post": 20}  # 裸地の田: 時期の NDVI のこの百分位 ≤ 0.35（春は麦を除く・秋はひこばえがあっても刈った後の田を残す）
ERA = "ECMWF/ERA5_LAND/HOURLY"
NAMES = ["A", "N", "Ad", "Nd", "W", "Nw"]
SC = {"s2": [100, 1, 100, 1, 1, 1], "s1": [10, 1, 10, 1, 1, 1]}   # int16 で受け取るための倍率
SIG = {"s2d": 0.08, "s2": 0.08, "r": 1.0}                            # 指数にするときの目盛り（県全体の田の標準偏差に合わせ直す）
WI_W = {"s2d": 0.35, "s2": 0.15, "r": 0.25, "ww": 0.25}
MIN_N = {"s2": 6, "s2d": 3, "r": 10, "rd": 5, "ww": 6}
CHUNK = 3                         # 1回の computePixels で取る時期の数（失敗したら1時期ずつ）


# ---------------- 時期 ----------------
def season_dates(k):
    y, kind = int(k[:4]), k[4:]
    a, b = [x for t in SEAS if t[0] == kind for x in t[1:]]
    return f"{y}-{a}", (dt.date.fromisoformat(f"{y}-{b}") + dt.timedelta(days=1)).isoformat()


def ready_seasons(cfg, last):
    """ERA5-Land（雪・雨）が last の日まであるときに、作れる時期（新しい方から wet_seasons 個。
    それより古い時期でも、一度足したものは wet_sum.npz に残る）"""
    if not last:
        return []
    out = []
    for y in range(int(cfg.get("trend_from", 2022)), last.year + 1):
        for kind, _, b in SEAS:
            if dt.date.fromisoformat(f"{y}-{b}") + dt.timedelta(days=3) <= last:
                out.append(f"{y}{kind}")
    return out[-int(cfg.get("wet_seasons", 10)):]


# ---------------- タイル ----------------
def zgrid(bbox, z):
    """bbox を覆う、ズーム z のタイル画素そのままの格子（EPSG:3857）"""
    w, s, e, n = bbox
    size = 2 * ORIG / (TILE * 2 ** z)
    x0 = RM * math.radians(w); x1 = RM * math.radians(e)
    y0 = RM * math.log(math.tan(math.pi / 4 + math.radians(s) / 2)); y1 = RM * math.log(math.tan(math.pi / 4 + math.radians(n) / 2))
    X0 = int(math.floor((x0 + ORIG) / size)); X1 = int(math.ceil((x1 + ORIG) / size))
    Y0 = int(math.floor((ORIG - y1) / size)); Y1 = int(math.ceil((ORIG - y0) / size))
    return {"x0": X0 * size - ORIG, "y1": ORIG - Y0 * size, "px": size, "w": X1 - X0, "h": Y1 - Y0, "z": z, "X0": X0, "Y0": Y0}


class TileError(Exception):
    pass


def decode_dem(b):
    """標高PNG → m（欠測 NaN）。x = R·2^16 + G·2^8 + B, x<2^23: 0.01x, x=2^23: 欠測, x>2^23: 0.01(x−2^24)"""
    import numpy as np
    from PIL import Image
    a = np.asarray(Image.open(io.BytesIO(b)).convert("RGB"), np.int64)
    x = (a[..., 0] << 16) | (a[..., 1] << 8) | a[..., 2]
    return np.where(x == 1 << 23, np.nan, np.where(x > 1 << 23, x - (1 << 24), x) * 0.01).astype(np.float32)


def decode_rgba(b):
    import numpy as np
    from PIL import Image
    return np.asarray(Image.open(io.BytesIO(b)).convert("RGBA"), np.uint8)


class HttpTiles:
    """XYZ タイルを取る（Actions から）。同じ実行の中ではディスクに置いて使い回す（404 は空のファイル）。スレッドをまたいで 5件/秒 以下"""
    def __init__(self, cache, log):
        self.cache = cache; self.log = log; self.lock = threading.Lock(); self.last = 0.0; self.local = threading.local()
        self.fails = {}; self.stat = {"get": 0, "hit": 0, "404": 0, "err": 0}

    def _sess(self):
        import requests
        if not hasattr(self.local, "s"):
            self.local.s = requests.Session(); self.local.s.headers["User-Agent"] = UA
        return self.local.s

    def _wait(self):
        with self.lock:
            t = max(time.time(), self.last + MIN_GAP); self.last = t
        time.sleep(max(0.0, t - time.time()))

    def raw(self, kind, z, x, y):
        """PNG のバイト列。タイルがない（404）なら None。取れなければ TileError"""
        from urllib.parse import urlparse
        p = os.path.join(self.cache, kind, str(z), str(x), f"{y}.png")
        if os.path.exists(p):
            self.stat["hit"] += 1
            with open(p, "rb") as f:
                return f.read() or None
        last = "?"
        for url in URLS[kind]:
            host = urlparse(url).netloc
            if self.fails.get(host, 0) >= DOWN_AFTER:
                last = f"{host} は今回は使わない（続けて失敗）"; continue
            for k in range(4):
                self._wait(); self.stat["get"] += 1
                try:
                    r = self._sess().get(url.format(z=z, x=x, y=y), timeout=20)
                except Exception as e:                       # DNS・接続・時間切れ
                    last = f"{host}: {type(e).__name__}"; time.sleep(2 ** k); continue
                if r.status_code in (404, 204) or (r.status_code == 200 and not r.content):
                    self.fails[host] = 0; self.stat["404"] += 1; self._save(p, b""); return None
                if r.status_code == 200 and r.content[:4] == b"\x89PNG":
                    self.fails[host] = 0; self._save(p, r.content); return r.content
                last = f"{host}: HTTP {r.status_code}"
                if r.status_code != 429 and r.status_code < 500:
                    break                                     # 403 など: 待っても同じなので次のサーバーへ
                time.sleep(2 ** k)
            with self.lock:
                self.fails[host] = self.fails.get(host, 0) + 1
        self.stat["err"] += 1
        raise TileError(last)

    def _save(self, p, b):
        os.makedirs(os.path.dirname(p), exist_ok=True)
        tmp = f"{p}.{threading.get_ident()}.part"
        with open(tmp, "wb") as f:
            f.write(b)
        os.replace(tmp, p)

    def get(self, kind, z, x, y):
        b = self.raw(kind, z, x, y)
        if b is None:
            return None
        return decode_rgba(b) if kind == "soil" else decode_dem(b)


def mosaic(get, z, X0, Y0, w, h, fill, dtype, ch=0, need=None):
    """ズーム z・全体の画素の位置 (X0, Y0) から w×h の範囲をタイルで埋める。need（w×h の真偽）が一つもないタイルは取らない。
    戻り値 (配列, 取れたタイル数, 404 の数, 失敗の数)"""
    import numpy as np
    out = np.full((h, w, ch) if ch else (h, w), fill, dtype); ok = none = bad = 0
    for ty in range(Y0 // TILE, (Y0 + h - 1) // TILE + 1):
        for tx in range(X0 // TILE, (X0 + w - 1) // TILE + 1):
            gx0, gx1 = max(X0, tx * TILE), min(X0 + w, (tx + 1) * TILE); gy0, gy1 = max(Y0, ty * TILE), min(Y0 + h, (ty + 1) * TILE)
            if need is not None and not need[gy0 - Y0:gy1 - Y0, gx0 - X0:gx1 - X0].any():
                continue
            try:
                a = get(z, tx, ty)
            except TileError:
                bad += 1; continue
            if a is None:
                none += 1; continue
            out[gy0 - Y0:gy1 - Y0, gx0 - X0:gx1 - X0] = a[gy0 - ty * TILE:gy1 - ty * TILE, gx0 - tx * TILE:gx1 - tx * TILE]
            ok += 1
    return out, ok, none, bad


def to_grid_fn(g):
    import numpy as np
    def f(xy):
        x = RM * np.radians(xy[:, 0]); y = RM * np.log(np.tan(np.pi / 4 + np.radians(xy[:, 1]) / 2))
        return np.column_stack([(x - g["x0"]) / g["px"], (g["y1"] - y) / g["px"]])
    return f


def block_mean(A, ga, g):
    """格子 ga の値 A（NaN=なし）を、格子 g の各画素の範囲で平均する（NaN を除く）"""
    import numpy as np
    ce = np.clip(np.round((g["x0"] + np.arange(g["w"] + 1) * g["px"] - ga["x0"]) / ga["px"]).astype(int), 0, ga["w"])
    re = np.clip(np.round((ga["y1"] - (g["y1"] - np.arange(g["h"] + 1) * g["px"])) / ga["px"]).astype(int), 0, ga["h"])
    ok = ~np.isnan(A)
    SV = np.zeros((A.shape[0] + 1, A.shape[1] + 1)); SC_ = np.zeros_like(SV)
    SV[1:, 1:] = np.cumsum(np.cumsum(np.where(ok, A, 0), 0), 1); SC_[1:, 1:] = np.cumsum(np.cumsum(ok, 0), 1)
    def box(S):
        return S[re[1:, None], ce[None, 1:]] - S[re[:-1, None], ce[None, 1:]] - S[re[1:, None], ce[None, :-1]] + S[re[:-1, None], ce[None, :-1]]
    t, c = box(SV), box(SC_)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(c > 0, t / np.maximum(c, 1), np.nan).astype(np.float32)


def neighbor_parcels(data_dir, index, cid):
    x, y = map(int, cid.split("_")); ids = {c["id"] for c in index["cells"]}; out = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            c = f"{x + dx}_{y + dy}"
            if c in ids:
                out += (read_json(os.path.join(data_dir, "cells", c, "parcels.geojson")) or {"features": []})["features"]
    return {"features": out}


# ---------------- 土 ----------------
_SOIL = None
def soil_table():
    """[(r,g,b,記号,名前,軟らかさ|None)]、色キーの並びと添字"""
    import numpy as np
    global _SOIL
    if _SOIL is None:
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "soil_rgb.json"), encoding="utf-8") as f:
            t = json.load(f)
        keys = np.array([r << 16 | g << 8 | b for r, g, b, *_ in t], np.int64); order = np.argsort(keys)
        rgb = np.array([r[:3] for r in t], np.int64)
        _SOIL = {"t": t, "keys": keys[order], "idx": order, "rgb": rgb,
                 "score": np.array([np.nan if r[5] is None else r[5] for r in t], np.float32)}
    return _SOIL


def soil_decode(rgba):
    """RGBA → 表の添字（−1 = 透明・データなし, −2 = 表にない色）。戻り値 (添字, 色が一致した画素, 近い色で決めた画素)
    表の色は記号を表す番号のようなもの（となりの記号と 1〜2 しか違わない）なので、近い色で決めるのは次のときだけ:
      不透明（alpha 255）・黒に近い色（境界線や文字。R+G+B ≤ 12、泥炭土の色のそば）でも白に近い色（R+G+B ≥ 760）でもない・L1 距離 3 以内の候補がどれも
      同じくらいの軟らかさ（差 0.1 以内。一番近いものを使う）。残った表にない色（線・文字）は、3×3 のまわりで一番多い記号で埋める"""
    import numpy as np
    T = soil_table(); a = rgba.reshape(-1, 4).astype(np.int64)
    key = a[:, 0] << 16 | a[:, 1] << 8 | a[:, 2]
    pos = np.clip(np.searchsorted(T["keys"], key), 0, len(T["keys"]) - 1)
    hit = T["keys"][pos] == key
    out = np.where(hit, T["idx"][pos], -2)
    rgbsum = a[:, :3].sum(1)
    miss = ~hit & (a[:, 3] == 255) & (rgbsum > 12) & (rgbsum < 760)
    if miss.any():
        u, inv = np.unique(key[miss], return_inverse=True)
        uc = np.stack([u >> 16, (u >> 8) & 255, u & 255], 1)
        d = np.abs(uc[:, None, :] - T["rgb"][None, :, :]).sum(2); j = d.argmin(1)
        sc = np.where(np.isnan(T["score"]), -9, T["score"])[None, :]
        near = d <= 3
        hi = np.where(near, sc, -99).max(1); lo = np.where(near, sc, 99).min(1)
        okm = near.any(1) & (hi - lo <= 0.1 + 1e-6)
        out[miss] = np.where(okm, j, -2)[inv]
    out[a[:, 3] == 0] = -1
    sh = rgba.shape[:2]; out = out.reshape(sh)
    nearm = (miss & (out.ravel() >= 0)).reshape(sh)
    bad = (out == -2) & (rgba[..., 3] > 0)
    if bad.any():                                        # 線・文字: まわり 3×3 の一番多い記号（なければそのまま −2）
        P = np.pad(out, 1, constant_values=-1); r_, c_ = np.nonzero(bad)
        nb = np.stack([P[r_ + dy, c_ + dx] for dy in range(3) for dx in range(3)], 1)
        fill = np.full(len(r_), -2, np.int64)
        for i, row in enumerate(nb):
            v = row[row >= 0]
            if len(v):
                vals, cnt = np.unique(v, return_counts=True); fill[i] = vals[cnt.argmax()]
        out[r_, c_] = fill; nearm[r_, c_] |= fill >= 0
    return out, (hit & (a[:, 3] > 0)).reshape(sh), nearm


def soil_part(tiles, parcels, cell, log):
    """{"src","names","match","p"} または None（タイルが取れない）"""
    import numpy as np
    from trend import labels
    T = soil_table()
    for z in (15, 14):
        w, s_, e, n = cell["bbox"]; g = zgrid([w - 0.002, s_ - 0.002, e + 0.002, n + 0.002], z)   # セルからはみ出た田も
        lab = labels(parcels, g, edges=(0,))
        rgba, ok, none, bad = mosaic(lambda z_, x, y: tiles.get("soil", z_, x, y), z, g["X0"], g["Y0"], g["w"], g["h"], 0, np.uint8, 4, need=lab > 0)
        if bad > (ok + none + bad) / 2 or ok + none + bad == 0:
            log(f"{cell['id']}: 土壌図のタイルが取れません（z{z}: 取れた{ok}・なし{none}・失敗{bad}）"); return None
        if ok:
            break
    else:
        log(f"{cell['id']}: 土壌図のタイルがありません（範囲外）"); return None
    idx, hit, near = soil_decode(rgba)
    inp = lab > 0; nin = int(((rgba[..., 3] > 0) & inp).sum()) or 1; nhit = int((hit & inp).sum()); nnear = int((near & inp).sum())
    ls, vs = lab[inp], idx[inp]; o = np.argsort(ls, kind="stable"); ls, vs = ls[o], vs[o]
    cuts = np.searchsorted(ls, np.arange(1, len(parcels["features"]) + 2))
    p = {}; names = {}
    for k, f in enumerate(parcels["features"], 1):
        v = vs[cuts[k - 1]:cuts[k]]
        if not len(v):
            continue
        good = v[v >= 0]; sc = T["score"][good]; sc = sc[~np.isnan(sc)]
        nod = round(100 * (1 - len(sc) / len(v)))
        if len(good):
            u, c = np.unique(good, return_counts=True); dom = int(u[c.argmax()]); code = T["t"][dom][3]; names[code] = T["t"][dom][4]
            share = round(100 * c.max() / len(v))
        else:
            code = None; share = 0
        p[f["properties"]["pid"]] = [round(100 * float(sc.mean())) if len(sc) else None, code, nod, share]
    pe = np.array([r[3].startswith("B") for r in T["t"]] + [False, False])[vs]          # 泥炭土・黒泥土（−1・−2 は末尾の False）
    npe = int(pe.sum())
    if npe:
        log(f"{cell['id']}: 土 泥炭土・黒泥土の画素 {npe}（うち近い色で決めた {round(100 * int((pe & near[inp][o]).sum()) / npe)}%）")
    return {"src": f"naro-soil-inventory z{z}", "names": names,
            "match": [round(100 * nhit / nin, 1), round(100 * nnear / nin, 1)], "p": p}


# ---------------- 地形 ----------------
def dem_part(tiles, sat, parcels, around, cell, log):
    """(terr の中身, 10m の dz 配列（セルの格子, NaN=なし）)。取れなければ (None, None)"""
    import numpy as np
    from trend import labels
    w, s, e, n = cell["bbox"]; pb = [w - PAD_DEG[0], s - PAD_DEG[1], e + PAD_DEG[0], n + PAD_DEG[1]]
    gz = zgrid(pb, 15); g10 = grid_for(pb); gc = grid_for(cell["bbox"])
    H4 = np.full((gz["h"], gz["w"]), np.nan, np.float32); S4 = np.full(H4.shape, 255, np.uint8); tot = [0, 0, 0]
    for code, layer in ((0, "dem5a"), (1, "dem5b"), (1, "dem5c")):
        A, ok, none, bad = mosaic(lambda z, x, y: tiles.get(layer, z, x, y), 15, gz["X0"], gz["Y0"], gz["w"], gz["h"], np.nan, np.float32, need=np.isnan(H4))
        m = np.isnan(H4) & ~np.isnan(A); H4[m] = A[m]; S4[m] = code
        tot = [tot[0] + ok, tot[1] + none, tot[2] + bad]
        if not np.isnan(H4).any():
            break
    if np.isnan(H4).any():                              # 10m（z14）で残りを埋める（z15 の2×2画素に同じ値）
        X0, Y0 = gz["X0"] // 2, gz["Y0"] // 2; w14 = (gz["X0"] + gz["w"] - 1) // 2 - X0 + 1; h14 = (gz["Y0"] + gz["h"] - 1) // 2 - Y0 + 1
        xi = (gz["X0"] + np.arange(gz["w"])) // 2 - X0; yi = (gz["Y0"] + np.arange(gz["h"])) // 2 - Y0
        need14 = np.zeros((h14, w14), bool); need14[np.ix_(yi, xi)] |= np.isnan(H4)
        A, ok, none, bad = mosaic(lambda z, x, y: tiles.get("dem_png", z, x, y), 14, X0, Y0, w14, h14, np.nan, np.float32, need=need14)
        A = A[np.ix_(yi, xi)]; m = np.isnan(H4) & ~np.isnan(A); H4[m] = A[m]; S4[m] = 2
        tot = [tot[0] + ok, tot[1] + none, tot[2] + bad]
    if tot[0] == 0 or tot[2] > tot[0]:                  # タイルがほとんど取れない（海の欠測は NaN のままでよい）
        if sat is None:
            log(f"{cell['id']}: 標高タイルが取れません（取れた{tot[0]}・なし{tot[1]}・失敗{tot[2]}）"); return None, None
        log(f"{cell['id']}: 標高タイルが取れないので GEE の Copernicus DEM（30m）を使います（失敗{tot[2]}）")
        H10 = sat.dem(g10).astype(np.float32); H4 = None; srcname = "copernicus-glo30"
    else:
        H10 = block_mean(H4, gz, g10); srcname = "gsi-dem5a/5b/5c/dem10b"
    # 田の画素（周りのセルの田も）・この田の番号（10m, 縁を含む）
    paddy = labels(around, g10, edges=(0,)) > 0
    lab10 = labels(parcels, g10, edges=(0,))
    T = twi(H10)
    if H4 is not None:
        ground = gz["px"] * math.cos(math.radians((s + n) / 2))
        lab4 = labels(parcels, gz, edges=(EDGE_M / ground, EDGE_M / 2 / ground, 0))
        D4 = np.full(H4.shape, np.nan, np.float32)
        o = np.argsort(lab4.ravel(), kind="stable"); ls = lab4.ravel()[o]
        cuts = np.searchsorted(ls, np.arange(1, len(parcels["features"]) + 2))
    import shapely
    from shapely.geometry import shape
    tg = to_grid_fn(g10); yy, xx = np.mgrid[0:g10["h"], 0:g10["w"]]; cx = xx + 0.5; cy = yy + 0.5
    p = {}; cover = np.zeros(4)
    for k, f in enumerate(parcels["features"], 1):
        pid = f["properties"]["pid"]
        try:
            gg = shapely.transform(shape(f["geometry"]), tg).buffer(0)
            far = gg.buffer(PCT_M / 10.0, quad_segs=4)
            x0, y0, x1, y1 = far.bounds
            c0, c1, r0, r1 = max(0, math.floor(x0)), min(g10["w"], math.ceil(x1)), max(0, math.floor(y0)), min(g10["h"], math.ceil(y1))
            X, Y = cx[r0:r1, c0:c1], cy[r0:r1, c0:c1]; Hs = H10[r0:r1, c0:c1]; Ps = paddy[r0:r1, c0:c1]
            ins = lab10[r0:r1, c0:c1] == k
            if not ins.any():
                ins = shapely.contains_xy(gg, X, Y)
            fin = ~np.isnan(Hs)
            src = 3; low = rng = ldir = None; zmed = None
            if H4 is not None:
                pix = o[cuts[k - 1]:cuts[k]]
                if len(pix):
                    r_, c_ = np.unravel_index(pix, H4.shape); z = H4[r_, c_]; sv = S4[r_, c_]; good = ~np.isnan(z)
                    if good.sum() >= 3:
                        zmed = float(np.median(z[good])); src = int(np.bincount(sv[good], minlength=3)[:3].argmax())
                        if (sv[good] == 0).mean() >= IN_COVER and good.sum() >= 12:
                            low, rng, ldir, res = in_field(r_[good], c_[good], z[good])
                            D4[r_[good], c_[good]] = res
                            src = 0
                        elif src == 0:
                            src = 1
            if zmed is None:
                v = Hs[ins & fin]
                if not len(v):
                    continue
                zmed = float(np.median(v)); src = 3 if H4 is None else 2
            cover[src] += 1
            dist_ok = lambda rr: shapely.contains_xy(gg.buffer(rr / 10.0, quad_segs=4), X, Y) & ~ins & fin
            ring = dist_ok(REL_M)
            pool = ring & Ps if (ring & Ps).sum() >= 10 else ring
            rel = round(100 * (zmed - float(np.median(Hs[pool])))) if pool.sum() >= 10 else None
            pp = shapely.contains_xy(far, X, Y) & ~ins & fin & Ps
            pctl = round(100 * float((Hs[pp] < zmed).mean())) if pp.sum() >= 10 else None
            hr = dist_ok(HILL_M)
            hill = round(100 * float((Hs[hr] > zmed + HILL_DZ).mean())) if hr.sum() else 0
            tw = T[r0:r1, c0:c1][ins & fin]
            p[pid] = [round(100 * zmed), rel, pctl, low, ldir, rng, hill, round(10 * float(np.median(tw))) if len(tw) else None, src]
        except Exception as ex:                          # 形がこわれた区画があっても、そのセル全体を止めない
            log(f"{cell['id']}: 地形 {pid} を飛ばします（{ex}）")
            continue
    dz = block_mean(D4, gz, gc) if H4 is not None else None
    n_ = max(1, cover.sum())
    return {"src": srcname, "cover": [round(100 * cover[i] / n_) for i in range(4)], "p": p}, dz


def in_field(r, c, z):
    """田の中の低い所: 平面を当てはめて傾きを除き、中央値より LOW_M 以上低く LOW_MIN_PX 画素以上つながった所の割合（%）・
    凹凸（p95−p5, cm）・低い所の向き（8方位, 北=0。なければ −1）と、画素ごとの差（中央値との差, m）"""
    import numpy as np
    A = np.column_stack([np.ones(len(z)), c - c.mean(), r - r.mean()])
    coef, *_ = np.linalg.lstsq(A, z.astype(np.float64), rcond=None)
    res = z - A @ coef; res = res - np.median(res)
    rng = round(100 * float(np.percentile(res, 95) - np.percentile(res, 5)))
    lowm = res < -LOW_M
    keep = np.zeros(len(z), bool)
    if lowm.any():                                       # つながり（上下左右）を数える
        pos = {(int(a), int(b)): i for i, (a, b) in enumerate(zip(r, c)) if lowm[i]}
        seen = set()
        for st in pos:
            if st in seen:
                continue
            comp = [st]; seen.add(st); q = [st]
            while q:
                a, b = q.pop()
                for nb in ((a + 1, b), (a - 1, b), (a, b + 1), (a, b - 1)):
                    if nb in pos and nb not in seen:
                        seen.add(nb); comp.append(nb); q.append(nb)
            if len(comp) >= LOW_MIN_PX:
                keep[[pos[x] for x in comp]] = True
    share = round(100 * float(keep.mean()))
    d = -1
    if keep.any():
        dx = c[keep].mean() - c.mean(); dy = r[keep].mean() - r.mean()
        if math.hypot(dx, dy) >= 1:
            d = int(round((math.degrees(math.atan2(dx, -dy)) + 360) % 360 / 45)) % 8
    return share, rng, d, res.astype(np.float32)


def twi(H, cell=10.0):
    """地形の湿りやすさ ln(集水面積 / tanβ)（穴をうめてから D8 で流す）。範囲の端で集水域が切れるので、比べるのは相対値だけ"""
    import numpy as np, heapq
    h, w = H.shape; nan = np.isnan(H)
    Z = np.where(nan, np.nanmin(H) if (~nan).any() else 0, H).astype(np.float64)
    F = Z.copy(); closed = np.zeros((h, w), bool); hp = []
    nb8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    edge = np.zeros((h, w), bool); edge[0, :] = edge[-1, :] = edge[:, 0] = edge[:, -1] = True
    if nan.any():                                        # 欠測（海など）のとなりも出口にする
        m = nan.copy(); m[1:] |= nan[:-1]; m[:-1] |= nan[1:]; m[:, 1:] |= nan[:, :-1]; m[:, :-1] |= nan[:, 1:]; edge |= m & ~nan
    closed |= nan
    for i, j in zip(*np.nonzero(edge)):
        closed[i, j] = True; heapq.heappush(hp, (F[i, j], int(i), int(j)))
    Fl = F.tolist(); Cl = closed.tolist()
    while hp:
        zc, i, j = heapq.heappop(hp)
        for di, dj in nb8:
            a, b = i + di, j + dj
            if 0 <= a < h and 0 <= b < w and not Cl[a][b]:
                Cl[a][b] = True
                if Fl[a][b] <= zc:
                    Fl[a][b] = zc + 1e-5
                heapq.heappush(hp, (Fl[a][b], a, b))
    F = np.array(Fl)
    best = np.zeros((h, w)); down = np.full((h, w), -1, np.int64); idx = np.arange(h * w).reshape(h, w)
    P = np.pad(F, 1, constant_values=np.inf); I = np.pad(idx, 1, constant_values=-1)
    for di, dj in nb8:
        s = (F - P[1 + di:1 + di + h, 1 + dj:1 + dj + w]) / (cell * math.hypot(di, dj))
        m = s > best; best[m] = s[m]; down[m] = I[1 + di:1 + di + h, 1 + dj:1 + dj + w][m]
    acc = np.ones(h * w); dn = down.ravel()
    for i in np.argsort(-F.ravel(), kind="stable").tolist():
        if dn[i] >= 0:
            acc[dn[i]] += acc[i]
    gy, gx = np.gradient(np.where(nan, Z, H).astype(np.float64), cell)
    tanb = np.maximum(np.hypot(gx, gy), 0.002)
    out = np.log(acc.reshape(h, w) * cell / tanb); out[nan] = np.nan
    return out


# ---------------- 乾き（衛星） ----------------
class WetFail(Exception):
    def __init__(self, msg, fail):
        super().__init__(msg); self.fail = fail


def _fail(v):
    """失敗の記録 [回数, 最後に失敗した日]（前の形の回数だけも読む）"""
    return v if isinstance(v, list) else [int(v or 0), "2000-01-01"]


def fail_blocked(v, today):
    """2回続けて失敗し、まだ RETRY_DAYS 日たっていない時期（今は飛ばす）。日がたてば、直したコードでもう一度試す"""
    c, d = _fail(v)
    if c < 2:
        return False
    try:
        return (today - dt.date.fromisoformat(d)).days < RETRY_DAYS
    except ValueError:
        return False


def load_sums(path, shape, seasons):
    """wet_sum.npz（時期ごとの合計を足したもの。int32）。形・時期の一覧が wet.json と合わなければ空"""
    import numpy as np
    try:
        with np.load(path) as z:
            if sorted(str(x) for x in z["seasons"]) != sorted(seasons):
                return {}
            out = {k: z[k].astype(np.float64) for k in z.files if k != "seasons"}
        return out if all(v.shape == shape for v in out.values()) else {}
    except Exception:
        return {}


def save_sums(path, tot, seasons, mask):
    import numpy as np
    arr = {k: np.where(mask, np.round(v), 0).astype(np.int32) for k, v in tot.items()}   # 田の外は 0（よく縮む）
    tmp = path + ".part.npz"
    np.savez_compressed(tmp, seasons=np.array(sorted(seasons)), **arr)
    os.replace(tmp, path)


def wet_part(sat, parcels, cell, g, seasons, prev, log, today, sum_path, deadline=None):
    """(wet の中身 | None, wi 配列 | None, 作った時期があったか)。prev = 前回の wet。
    すでに足した時期（wet_sum.npz）はそのまま使い、足りない時期だけ GEE で取って足す（取れた分から保存するので、時間切れでも無駄にならない）。
    足りない時期がどれも取れなかったときは例外（コードの誤りなどで全セルがくり返し失敗するとき、run_all が早めにやめられるように）"""
    import numpy as np
    from trend import labels
    prev = prev or {}; fail = {k: _fail(v) for k, v in (prev.get("fail") or {}).items()}
    have = list(prev.get("seasons") or [])
    tot = load_sums(sum_path, (g["h"], g["w"]), have) if have else {}
    if have and not tot:
        log(f"{cell['id']}: 乾き 前回の合計（wet_sum.npz）がないので、はじめから作り直します"); have = []
    used = list(have); new = []
    inpaddy = labels(parcels, g, edges=(0,)) > 0
    pending = [k for k in seasons if k not in have and not fail_blocked(fail.get(k), today)]
    queue = [pending[i:i + CHUNK] for i in range(0, len(pending), CHUNK)]; nerr = 0
    while queue:
        if deadline and time.time() > deadline:
            log(f"{cell['id']}: 乾き 時間切れ（次回に続けます）"); break
        ch = queue.pop(0); t0 = time.time()
        try:
            arr = sat.sums(cell["bbox"], g, ch)
        except Exception as e:
            if len(ch) > 1:
                log(f"{cell['id']}: 乾き {ch} を1時期ずつに分けます（{str(e)[:120]}）"); queue = [[k] for k in ch] + queue; continue
            c = _fail(fail.get(ch[0]))[0] + 1; fail[ch[0]] = [c, today.isoformat()]; nerr += 1
            log(f"{cell['id']}: 乾き {ch[0]} 失敗 {c}回目（{str(e)[:160]}）"); continue
        for k in ch:
            for sn in ("s2", "s1"):
                for b in NAMES:
                    a = arr.get(f"{sn}_{k}_{b}")
                    if a is not None:
                        key = f"{sn}_{b}"; tot[key] = tot.get(key, 0) + a.astype(np.float64)
            used.append(k); new.append(k); fail.pop(k, None)
        save_sums(sum_path, tot, used, inpaddy)
        log(f"{cell['id']}: 乾き {','.join(ch)}（{time.time() - t0:.0f}秒）")
    if pending and not new and nerr:
        raise WetFail(f"乾き: 足りない {len(pending)} 時期がどれも取れませんでした", fail)
    if not new and have and prev.get("p") is not None:
        return None, None, fail                          # 足した時期がない（前回のまま）
    if not used:
        return None, None, fail
    wi, comp = wet_index(tot, (g["h"], g["w"]))
    wi[~inpaddy] = np.nan                                 # 田の外は保存しない（表示は田の中だけ）
    lab = labels(parcels, g)
    p = {}
    o = np.argsort(lab.ravel(), kind="stable"); ls = lab.ravel()[o]; cuts = np.searchsorted(ls, np.arange(1, len(parcels["features"]) + 2))
    nobs = tot.get("s2_N", np.zeros(wi.shape)) + tot.get("s1_N", np.zeros(wi.shape))
    for k, f in enumerate(parcels["features"], 1):
        pix = o[cuts[k - 1]:cuts[k]]
        if not len(pix):
            continue
        v = wi.ravel()[pix]; v = v[~np.isnan(v)]
        if len(v) < 2:
            continue
        def med(key, sc):
            x = comp[key].ravel()[pix]; x = x[~np.isnan(x)]
            return round(float(np.median(x)) * sc) if len(x) else None
        p[f["properties"]["pid"]] = [round(100 * float(np.median(v))), round(100 * float(np.percentile(v, 90))), round(100 * float((v > 1).mean())),
                                     med("s2d", 1000), med("r", 10), med("ww", 100), int(np.median(nobs.ravel()[pix]))]
    return {"src": "s2-b12+s1-vv vs 6km cropland (esa-worldcover), era5-land hourly snow/rain", "seasons": sorted(used), "fail": fail, "p": p}, wi, fail


def wet_index(tot, shape):
    """時期を合わせた合計から、画素ごとの成分（s2, s2d, r, ww）と指数 wi（重みつき平均。ない成分は除いて重みを割り直す）"""
    import numpy as np
    z = np.zeros(shape); T = lambda k: tot.get(k, z)
    def mean(a, n, sc, mn):
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(n >= mn, a / sc / np.maximum(n, 1), np.nan)
    s2 = mean(T("s2_A"), T("s2_N"), 100, MIN_N["s2"]); s2d = mean(T("s2_Ad"), T("s2_Nd"), 100, MIN_N["s2d"])
    r = mean(T("s1_A"), T("s1_N"), 10, MIN_N["r"]); rd = mean(T("s1_Ad"), T("s1_Nd"), 10, MIN_N["rd"])
    r = np.where(np.isnan(rd), r, rd)
    nw = T("s2_Nw") + T("s1_Nw")
    with np.errstate(invalid="ignore", divide="ignore"):
        ww = np.where(nw >= MIN_N["ww"], (T("s2_W") + T("s1_W")) / np.maximum(nw, 1), np.nan)
    comp = {"s2": s2, "s2d": s2d, "r": r, "ww": ww}
    zs = {"s2d": s2d / SIG["s2d"], "s2": s2 / SIG["s2"], "r": r / SIG["r"], "ww": (ww - 0.05) / 0.08}
    num = np.zeros(shape); den = np.zeros(shape)
    for k, v in zs.items():
        ok = ~np.isnan(v); num[ok] += WI_W[k] * v[ok]; den[ok] += WI_W[k]
    with np.errstate(invalid="ignore", divide="ignore"):
        wi = np.where(den >= 0.25, num / np.maximum(den, 1e-9), np.nan)
    return wi, comp


def enc_wi(wi):
    import numpy as np
    out = np.zeros(wi.shape, np.uint8); ok = ~np.isnan(wi)
    out[ok] = np.clip(np.round((np.clip(wi[ok], -3, 3) + 3) / 6 * 254) + 1, 1, 255).astype(np.uint8); return out


def enc_dz(dz):
    import numpy as np
    out = np.zeros(dz.shape, np.uint8); ok = ~np.isnan(dz)
    out[ok] = np.clip(np.round(128 + dz[ok] / 0.02), 1, 255).astype(np.uint8); return out


class GEEWet:
    """乾き（Sentinel-2 SWIR・Sentinel-1 VV・水）の時期ごとの合計を GEE で作る。backend は update_ndvi.GEEBackend, px は pixels.GEEPixels"""
    def __init__(self, backend, px):
        self.b = backend; self.px = px; self.ee = backend.ee

    def era5_last(self, today):
        ee = self.ee; t = time.time() * 1000
        c = ee.ImageCollection(ERA).filterDate(ee.Date(t).advance(-150, "day"), ee.Date(t).advance(1, "day"))
        v = with_retry(lambda: c.aggregate_max("system:time_start").getInfo())
        return dt.datetime.fromtimestamp(v / 1000, dt.timezone.utc).date() if v else None

    def dem(self, g):
        import numpy as np
        ee = self.ee
        img = ee.ImageCollection("COPERNICUS/DEM/GLO30").select("DEM").map(lambda i: i.resample("bilinear")).mosaic().rename("h")
        a = np.asarray(with_retry(lambda: ee.data.computePixels(self.px._req(img.unmask(-9999).toFloat(), g)))["h"], np.float32)
        a[a < -1000] = np.nan
        return a

    def _box(self, bbox):
        cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
        return [cx - REF_DEG[0], cy - REF_DEG[1], cx + REF_DEG[0], cy + REF_DEG[1]]

    def _num(self, x, fill):
        ee = self.ee
        return ee.Number(ee.Algorithms.If(ee.Algorithms.IsEqual(x, None), fill, x))

    def _era(self, t, rect):
        """観測の時刻 t（ms）の前 RAIN_H 時間の雨（mm）・前24時間の積雪の最大（m）・前12時間の地面の温度の最低（K）。
        セルの中心の ERA5-Land の格子（約9km）の値。日の区切り（UTC）に関係なく、レーダーの夜の観測の直前の雨も数える"""
        ee = self.ee; e = ee.ImageCollection(ERA); t = ee.Date(t)
        def band(col, b):
            return ee.ImageCollection([ee.Image.constant(0).toDouble().rename(b).updateMask(0)]).merge(col.select([b]))
        img = ee.Image.cat(band(e.filterDate(t.advance(-RAIN_H, "hour"), t), "total_precipitation_hourly").sum().multiply(1000).rename("p"),
                           band(e.filterDate(t.advance(-24, "hour"), t.advance(1, "hour")), "snow_depth").max().rename("sd"),
                           band(e.filterDate(t.advance(-12, "hour"), t.advance(1, "hour")), "soil_temperature_level_1").min().rename("st"))
        v = ee.Dictionary(img.reduceRegion(ee.Reducer.first(), rect.centroid(100), 11132))
        return {"p": self._num(v.get("p"), 999), "sd": self._num(v.get("sd"), 0), "st": self._num(v.get("st"), 300)}

    def _crop(self):
        return self.ee.ImageCollection("ESA/WorldCover/v200").first().select("Map").eq(40)

    def _medcnt(self):
        ee = self.ee
        return ee.Reducer.median().combine(ee.Reducer.count(), "", True)

    def _s2(self, box, a, b, post):
        """(合計の画像 6バンド, 裸地（時期の NDVI の BARE_PCT 百分位 ≤ 0.35）の画像, 日ごとの画像の集まり)"""
        ee = self.ee; B = self.b; thr = B.cfg.get("cloud_score_min", 0.6)
        rect = ee.Geometry.Rectangle(box); crop = self._crop(); dem = ee.Image("USGS/SRTMGL1_003")
        def prep(i):
            r = i.select(["B3", "B4", "B8"]).divide(1e4).addBands(i.select(["B11", "B12"]).divide(1e4).resample("bilinear"))
            scl = i.select("SCL"); nd = r.normalizedDifference(["B8", "B4"]); mn = r.normalizedDifference(["B3", "B11"])
            snow = mn.gt(0.4).And(r.select("B3").gt(0.15))
            # 山の影（冬の低い太陽）は SWIR が暗くなって湿って見えるので除く
            lit = ee.Terrain.hillShadow(dem, i.getNumber("MEAN_SOLAR_AZIMUTH_ANGLE"), i.getNumber("MEAN_SOLAR_ZENITH_ANGLE"), 100).focalMin(2)
            cf = i.select("cs_cdf").gte(thr).And(lit)                  # 雲・影のない画素（SCL で雪を除く前）
            ok = cf.And(scl.remap([0, 1, 3, 8, 9, 10, 11], [0] * 7, 1))
            v = ok.And(nd.lt(NDVI_BARE)).And(snow.Not())
            wat = ok.And(mn.gt(0)).And(r.select("B8").lt(0.15)).And(snow.Not())
            sn = snow.Or(scl.eq(11)).And(cf)                            # 雪: 自分の判定と SCL の雪のどちらか（雲のない所で）
            # L・nd は使える画素だけ。ほかは 0/1 のまま残す（雪・雲の割合を数えるため）
            return (ee.Image.cat(r.select("B12").max(1e-3).log().updateMask(ok).rename("L"), v.rename("v"), wat.rename("w"), sn.rename("sn"),
                                 ok.rename("ok"), cf.rename("cf"), nd.updateMask(ok).rename("nd"))
                    .set({"day": ee.Date(i.get("system:time_start")).format("YYYY-MM-dd"), "system:time_start": i.get("system:time_start")}))
        col = (B._raw(box, a, b, "s2").filter(ee.Filter.lt("MEAN_SOLAR_ZENITH_ANGLE", MAX_ZEN))
               .linkCollection(B._csp(box, a, b), ["cs_cdf"]).map(prep))
        def per_day(ds):
            ds = ee.String(ds); dc = col.filter(ee.Filter.eq("day", ds)); img = dc.mosaic(); era = self._era(dc.aggregate_max("system:time_start"), rect)
            v = img.select("v").unmask(0); w = img.select("w").unmask(0); okb = img.select("ok").unmask(0); sn = img.select("sn").unmask(0)
            cfb = img.select("cf").unmask(0)
            L = img.select("L")
            M = L.updateMask(v.And(w.Not()).And(crop)).reduceRegion(self._medcnt(), rect, REF_SCALE, bestEffort=True, tileScale=2)
            fr = ee.Image.cat(okb.rename("c"), w.rename("w"), sn.rename("sn"), cfb.rename("cf")).updateMask(crop).reduceRegion(
                ee.Reducer.mean(), rect, REF_SCALE, bestEffort=True, tileScale=2)
            cnt = self._num(M.get("L_count"), 0); clr = self._num(fr.get("c"), 0)
            wf = self._num(fr.get("w"), 0).divide(clr.max(1e-3)); sf = self._num(fr.get("sn"), 0).divide(self._num(fr.get("cf"), 0).max(1e-3))
            okd = cnt.gte(MIN_REF).And(clr.gte(THR_CLR)).And(sf.lte(THR_SNOW)).And(era["sd"].lte(SNOW_M["s2"]))
            dry = ee.Image.constant(era["p"].lt(DRY_MM))
            wel = ee.Image.constant(wf.lte(THR_WF).And(ee.Number(1) if post else ds.slice(5).compareTo(WATER_LAST).lt(0)))
            an = ee.Image.constant(self._num(M.get("L_median"), 0)).subtract(L).clamp(-C2, C2).multiply(v).unmask(0)
            return (ee.Image.cat(an, v, an.multiply(dry), v.multiply(dry), w.multiply(v).multiply(wel), v.multiply(wel))
                    .rename(NAMES).toFloat().multiply(ee.Image.constant(okd))
                    .set({"day": ds, "ok": okd, "cnt": cnt, "clr": clr, "sf": sf, "wf": wf, "p": era["p"], "sd": era["sd"]}))
        days = ee.ImageCollection(col.aggregate_array("day").distinct().map(per_day))
        zero = ee.Image.constant([0] * 6).rename(NAMES).toFloat()
        nd = ee.ImageCollection([ee.Image.constant(0).toFloat().rename("nd").updateMask(0)]).merge(
            col.map(lambda i: i.select("nd").updateMask(i.select("sn").unmask(0).Not()).toFloat()))
        q = BARE_PCT["post" if post else "pre"]
        bare = nd.reduce(ee.Reducer.percentile([q])).lte(NDVI_BARE)
        return ee.ImageCollection([zero]).merge(days).sum(), bare, days

    def _s1(self, box, a, b, post, bare):
        ee = self.ee; rect = ee.Geometry.Rectangle(box); crop = self._crop()
        def prep(i):
            vv = i.select("VV"); vh = i.select("VH")
            vvs = ee.Image(10).pow(vv.divide(10)).focalMean(1, "square", "pixels").log10().multiply(10)
            val = vv.gt(-30).And(bare.unmask(vh.subtract(vv).lte(-7.5)))   # 麦など緑のある田は除く（S2 がない所は VH−VV で）
            return (ee.Image.cat(vvs.rename("S"), val.rename("v")).updateMask(vv.mask())
                    .set({"k": ee.Date(i.get("system:time_start")).format("YYYY-MM-dd").cat("_").cat(ee.String(i.get("orbitProperties_pass"))),
                          "system:time_start": i.get("system:time_start")}))
        col = (ee.ImageCollection("COPERNICUS/S1_GRD").filterBounds(rect).filterDate(a, b).filter(ee.Filter.eq("instrumentMode", "IW"))
               .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
               .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH")).map(prep))
        def per_k(k):
            k = ee.String(k); dc = col.filter(ee.Filter.eq("k", k)); img = dc.mosaic(); era = self._era(dc.aggregate_max("system:time_start"), rect)
            S = img.select("S"); v = img.select("v").unmask(0)
            M = S.updateMask(v.And(S.gt(-18)).And(crop)).reduceRegion(self._medcnt(), rect, REF_SCALE, bestEffort=True, tileScale=2)
            med = self._num(M.get("S_median"), 0); cnt = self._num(M.get("S_count"), 0)
            wat = S.lt(-18).And(S.subtract(med).lt(-4)).And(v).unmask(0)
            fr = ee.Image.cat(v.rename("c"), wat.rename("w")).updateMask(crop).reduceRegion(ee.Reducer.mean(), rect, REF_SCALE, bestEffort=True, tileScale=2)
            clr = self._num(fr.get("c"), 0); wf = self._num(fr.get("w"), 0).divide(clr.max(1e-3))
            okd = cnt.gte(MIN_REF).And(clr.gte(THR_CLR)).And(era["sd"].lte(SNOW_M["s1"])).And(era["st"].gte(FROZEN_K))
            dry = ee.Image.constant(era["p"].lt(DRY_MM))
            wel = ee.Image.constant(wf.lte(THR_WF).And(ee.Number(1) if post else k.slice(5, 10).compareTo(WATER_LAST).lt(0)))
            n1 = v.And(wat.Not()); an = S.subtract(med).clamp(-C1, C1).multiply(n1).unmask(0)
            return (ee.Image.cat(an, n1, an.multiply(dry), n1.multiply(dry), wat.multiply(wel), v.multiply(wel))
                    .rename(NAMES).toFloat().multiply(ee.Image.constant(okd))
                    .set({"day": k, "ok": okd, "cnt": cnt, "clr": clr, "wf": wf, "p": era["p"], "sd": era["sd"], "st": era["st"]}))
        days = ee.ImageCollection(col.aggregate_array("k").distinct().map(per_k))
        zero = ee.Image.constant([0] * 6).rename(NAMES).toFloat()
        return ee.ImageCollection([zero]).merge(days).sum(), days

    def _season(self, bbox, k):
        a, b = season_dates(k); post = k.endswith("post"); box = self._box(bbox)
        s2, bare, d2 = self._s2(box, a, b, post)
        s1, d1 = self._s1(box, a, b, post, bare)
        return s2, s1, d2, d1

    def sums(self, bbox, g, seasons):
        import numpy as np
        ee = self.ee; imgs = []
        for k in seasons:
            s2, s1, _, _ = self._season(bbox, k)
            for sn, im in (("s2", s2), ("s1", s1)):
                imgs.append(im.multiply(ee.Image.constant(SC[sn])).round().toInt16().rename([f"{sn}_{k}_{n}" for n in NAMES]))
        arr = with_retry(lambda: ee.data.computePixels(self.px._req(ee.Image.cat(imgs), g)))
        return {n: np.asarray(arr[n]) for n in arr.dtype.names}

    def probe(self, bbox, k):
        """その時期の日ごとの判定（使う・使わない、その理由の数字）"""
        ee = self.ee; _, _, d2, d1 = self._season(bbox, k); out = {}
        for sn, d in (("s2", d2), ("s1", d1)):
            keys = ["day", "ok", "cnt", "clr", "wf", "p", "sd"] + (["sf"] if sn == "s2" else ["st"])
            fc = d.map(lambda i: ee.Feature(None, i.toDictionary(keys)))
            out[sn] = [f["properties"] for f in with_retry(lambda: fc.getInfo())["features"]]
        return out


class FakeTiles:
    """動作確認用のタイル（ネットにつながない）。土: セルの西半分がグライ低地土・東半分が灰色低地土・丸く泥炭土（縁の色を少しずらす）。
    標高: 西ほど低い傾き・南北の低い溝・東の丘、5つに1つのタイルは dem5a がない（dem5b で埋まる）"""
    stat = {}
    def get(self, kind, z, x, y):
        import numpy as np
        T = soil_table()
        n = 2 ** z; yy, xx = np.mgrid[0:TILE, 0:TILE]
        lon = (x + (xx + 0.5) / TILE) / n * 360 - 180
        lat = np.degrees(np.arctan(np.sinh(np.pi * (1 - 2 * (y + (yy + 0.5) / TILE) / n))))
        fx = (lon * 40) % 1; fy = (lat * 60) % 1                    # セル（0.025°×0.01667°）の中の位置 0〜1
        if kind == "soil":
            rgb = {c: T["t"][i][:3] for i, c in enumerate(r[3] for r in T["t"])}
            a = np.zeros((TILE, TILE, 4), np.uint8); a[..., 3] = 255
            a[..., :3] = np.where((fx < 0.5)[..., None], rgb["F2"], rgb["F3"])
            a[((fx - 0.3) ** 2 + (fy - 0.3) ** 2 < 0.02), :3] = rgb["B1"]
            a[(xx % 64 == 0), 0] = np.minimum(255, a[(xx % 64 == 0), 0].astype(int) + 1)   # ぼかしのような1だけずれた色
            return a
        if kind == "dem5a" and (x + y) % 5 == 0:
            return None
        if kind not in ("dem5a", "dem5b", "dem_png"):
            return None
        h = 10 + 2.0 * fx + np.where(np.abs(fx - 0.6) < 0.03, -0.8, 0) + np.where(fx > 0.85, (fx - 0.85) * 60, 0)
        h = h - 0.25 * np.exp(-(((lon * 1000) % 1 - 0.5) ** 2 + ((lat * 1200) % 1 - 0.5) ** 2) / 0.02)   # 田の中の小さな凹み
        return (h + np.random.default_rng(x * 7919 + y).normal(0, 0.02, h.shape)).astype(np.float32)


class FakeSat:
    """動作確認用の乾き: セルの南西4分の1に湿った円（SWIR 暗い・レーダー強い・ときどき水）、ほかは小さなばらつき"""
    def era5_last(self, today):
        return today - dt.timedelta(days=6)

    def dem(self, g):
        import numpy as np
        return np.full((g["h"], g["w"]), 10, np.float32)

    def sums(self, bbox, g, seasons):
        import numpy as np
        yy, xx = np.mgrid[0:g["h"], 0:g["w"]]
        disc = (xx - g["w"] * 0.3) ** 2 + (yy - g["h"] * 0.7) ** 2 < (min(g["w"], g["h"]) * 0.22) ** 2
        out = {}
        for k in seasons:
            rng = np.random.default_rng(sum(map(ord, k)) + int(abs(bbox[0]) * 1000))
            n2 = np.full(disc.shape, 8.0); n1 = np.full(disc.shape, 12.0)
            a2 = np.where(disc, 0.15, 0) + rng.normal(0, 0.03, disc.shape); a1 = np.where(disc, 1.5, 0) + rng.normal(0, 0.4, disc.shape)
            vals = {"s2": [a2 * n2 * 100, n2, a2 * 4 * 100, n2 * 0 + 4, np.where(disc, 2, 0), n2],
                    "s1": [a1 * n1 * 10, n1, a1 * 6 * 10, n1 * 0 + 6, np.where(disc, 3, 0), n1]}
            for sn, vs in vals.items():
                for b, v in zip(NAMES, vs):
                    out[f"{sn}_{k}_{b}"] = np.round(v).astype(np.int16)
        return out


class Sources:
    def __init__(self, tiles, sat):
        self.tiles = tiles; self.sat = sat


def sources(cfg, backend, gee_backend=None, px=None, log=print):
    if backend == "fake":
        return Sources(FakeTiles(), FakeSat() if cfg.get("wet_sat", True) else None)
    cache = os.path.join(os.environ.get("RUNNER_TEMP") or ".cache", "tiles")
    sat = GEEWet(gee_backend, px) if gee_backend is not None and cfg.get("wet_sat", True) else None
    return Sources(HttpTiles(cache, log), sat)


# ---------------- セルごと ----------------
def _retry_due(part, today):
    if part is None:
        return True
    if part.get("p") is not None and part.get("src") != "copernicus-glo30":   # GEE の 30m で代わりにした地形も、あとで標高タイルを試し直す
        return False
    try:
        return (today - dt.date.fromisoformat(part.get("tried", "2000-01-01"))).days >= RETRY_DAYS
    except ValueError:
        return True


def todo(m, today, ready, parts=("soil", "terr", "wet")):
    """作る部分の一覧（'soil', 'terr', 'wet'。parts のうち）"""
    if m.get("v") != VERSION:
        out = ["soil", "terr", "wet"] if ready else ["soil", "terr"]
    else:
        out = [k for k in ("soil", "terr") if _retry_due(m.get(k), today)]
        w = m.get("wet") or {}; fail = w.get("fail", {})
        done = set(w.get("seasons", [])) | {k for k, v in fail.items() if fail_blocked(v, today)}
        if ready and not set(ready) <= done:
            out.append("wet")
    return [k for k in out if k in parts]


def need(data_dir, cell, today, ready, parts=("soil", "terr", "wet")):
    return bool(todo(read_json(os.path.join(data_dir, "cells", cell["id"], "wet.json")) or {}, today, ready, parts))


def update_cell(src, cfg, data_dir, index, cell, today, ready, log, deadline=None, parts=("soil", "terr", "wet")):
    """1セルの wet.json / wet.png（と乾きの合計 wet_sum.npz）を、足りない部分だけ作る。
    保存した回数を返す（試した日・失敗の記録も公開し直して残すため、取れなかったときも数える）"""
    cdir = os.path.join(data_dir, "cells", cell["id"])
    parcels = read_json(os.path.join(cdir, "parcels.geojson"))
    if not parcels or not parcels.get("features"):
        return 0
    g = grid_for(cell["bbox"])
    sdir = cfg.get("wet_state_dir") or cdir      # 乾きの合計は公開しない（Pages の容量のため）。Actions では wet-state ブランチに置く
    os.makedirs(sdir, exist_ok=True)
    meta_p, png_p = os.path.join(cdir, "wet.json"), os.path.join(cdir, "wet.png")
    sum_p = os.path.join(sdir, f"{cell['id']}.npz") if cfg.get("wet_state_dir") else os.path.join(cdir, "wet_sum.npz")
    m = read_json(meta_p) or {}
    if m.get("v") != VERSION or any(m.get(k) != g[k] for k in g):
        m = {}
    todo_ = todo(m, today, ready if src.sat else [], parts)
    if not todo_:
        return 0
    layers = read_layers(png_p, {"dates": m.get("layers", []), "h": g["h"]}) if m and m.get("layers") else {}
    m.update({"v": VERSION, **g}); made = 0
    def save():
        keys = [k for k in ("wi", "dz") if k in layers]
        if keys:
            write_png(png_p, [layers[k] for k in keys])
        elif os.path.exists(png_p):
            os.remove(png_p)
        m["layers"] = keys
        write_json(meta_p, m)
    if "soil" in todo_:
        try:
            s = soil_part(src.tiles, parcels, cell, log)
        except Exception as e:
            log(f"{cell['id']}: 土 エラー {e}"); s = None
        m["soil"] = {**(s or {"src": None, "p": None}), "tried": today.isoformat()}; made += 1; save()
    if "terr" in todo_ and not (deadline and time.time() > deadline):
        around = neighbor_parcels(data_dir, index, cell["id"])
        try:
            t, dz = dem_part(src.tiles, src.sat, parcels, around, cell, log)
        except Exception as e:
            log(f"{cell['id']}: 地形 エラー {e}"); t, dz = None, None
        m["terr"] = {**(t or {"src": None, "p": None}), "tried": today.isoformat()}
        if dz is not None and (~__import__("numpy").isnan(dz)).any():
            layers["dz"] = enc_dz(dz)
        else:
            layers.pop("dz", None)
        made += 1; save()
    if "wet" in todo_ and src.sat and not (deadline and time.time() > deadline):
        prev = m.get("wet")
        try:
            w, wi, fail = wet_part(src.sat, parcels, cell, g, ready, prev, log, today, sum_p, deadline)
        except WetFail as e:
            m["wet"] = {**(m.get("wet") or {"seasons": [], "p": None}), "fail": e.fail}; save()
            raise
        if w is not None:
            m["wet"] = w
            if w["p"]:
                layers["wi"] = enc_wi(wi)
            else:
                layers.pop("wi", None)
            made += 1
        elif fail != (prev or {}).get("fail"):
            m["wet"] = {**(m.get("wet") or {"seasons": [], "p": None}), "fail": fail}; made += 1
        if not os.path.exists(sum_p) and (m.get("wet") or {}).get("seasons"):
            m["wet"]["seasons"] = []                     # 合計がなければ、時期はまだ済んでいない
        save()
    s, t, w = m.get("soil") or {}, m.get("terr") or {}, m.get("wet") or {}
    log(f"{cell['id']}: ハマりやすさ 土{'○' if s.get('p') else '×'}"
        + (f"（色一致 {s['match'][0]}%・近い色 {s['match'][1]}%）" if s.get("match") else "")
        + f" 地形{'○' if t.get('p') else '×'}" + (f"（{t.get('src')}, 5m/5b・5c/10m/30m {t.get('cover')}）" if t.get("p") else "")
        + f" 乾き{len(w.get('seasons', []))}時期（区画{len(parcels['features'])}）")
    return made


# ---------------- 確かめ（GEE） ----------------
def main():
    import argparse, sys
    from common import load_config
    ap = argparse.ArgumentParser(description="ハマりやすさの手がかりの確かめ（--probe: GEE で日ごとの判定を出す）")
    ap.add_argument("--probe", action="store_true"); ap.add_argument("--cells", required=True); ap.add_argument("--seasons")
    args = ap.parse_args()
    cfg = load_config(); data_dir = os.path.join(cfg["site_dir"], "data"); index = read_json(os.path.join(data_dir, "index.json"))
    from update_ndvi import GEEBackend
    from pixels import GEEPixels
    b = GEEBackend(cfg); sat = GEEWet(b, GEEPixels(b)); today = dt.datetime.now(dt.timezone.utc).date()
    last = sat.era5_last(today); ready = ready_seasons(cfg, last)
    print("ERA5-Land の最後の日:", last, "作れる時期:", ready)
    bands = with_retry(lambda: b.ee.ImageCollection(ERA).first().bandNames().getInfo())
    print("ERA5 のバンド（使うもの）:", {n: n in bands for n in ("total_precipitation_hourly", "snow_depth", "soil_temperature_level_1")})
    for cid in args.cells.split(","):
        cell = next(c for c in index["cells"] if c["id"] == cid); g = grid_for(cell["bbox"])
        for k in (args.seasons.split(",") if args.seasons else ready[-2:]):
            t0 = time.time(); pr = sat.probe(cell["bbox"], k)
            for sn, rows in pr.items():
                ok = [r for r in rows if r.get("ok")]
                print(f"{cid} {k} {sn}: {len(ok)}/{len(rows)} 日を使う（{time.time() - t0:.0f}秒）")
                for r in rows:
                    print("   ", json.dumps(r, ensure_ascii=False))
            t0 = time.time(); arr = sat.sums(cell["bbox"], g, [k])
            print(f"{cid} {k}: computePixels {time.time() - t0:.0f}秒", {n: int(a.sum()) for n, a in arr.items() if n.endswith(("_N", "_W"))})
    sys.stdout.flush()


if __name__ == "__main__":
    main()
