"""田の形（変形田かどうか）の手がかりを作る。生育傾向マップの圃場の評価に使う。

機械作業1回の時間を、筆ポリゴンの形から次の式で見積もる（作業幅 W・速度 v・旋回 t_t・切り返し t_b・手作業 c_m）:
  T = A/(W·v) + N·t_t + Σ t_b·n(α_j) + Σ A_m(α_j)/c_m
    N      = 行程の数。作業の向きごとに、幅 W 間隔の線で田を切って、線が田の中を通る区間（3m以上）を数える
             （凹んだ所で行程が分かれると、その分増える）。向きは田の辺（と外接長方形）に平行な向きから、N が一番少ないもの
    n(α)   = 内角 α の隅での切り返しの回数: 90°以下は 90/α 回、90〜150° は (150−α)/60 回、150° 以上は 0
    A_m(α) = 機械が入れず手作業になる隅の面積の下限 W²/(4·tan(α/2))
同じ広さの 1:3 の長方形（短辺 B=√(A/3)、隅は直角4つ）の T と比べた増え方を「形のロス」とする。
畦畔（周長）も、同じ長方形の周長 8B と比べる。面積が小さいことの影響は「広さ」で別に見るので、ここでは形だけを見る。

cells/<id>/shape.json: {"v":1, "W":…, "p":{"<pid>":[lbar, loss, perim, nacute, minang, dir]}}
  lbar = 平均行程長（m）, loss = 形のロス（%）, perim = 周長の倍率（%）,
  nacute = 75°未満の鋭角の隅の数, minang = 一番鋭い隅の角度（°）, dir = 作業の向き（北から時計回り °, 0〜179）
"""
import os, math
from common import write_json, read_json

VERSION = 1
W, V, T_TURN, T_BACK, C_MAN = 2.4, 1.0, 15.0, 20.0, 60.0   # 作業幅 m・速度 m/s・旋回 s/回・切り返し s/回・手作業 s/m²
MIN_PASS = 3.0            # これより短い区間は行程にしない（手作業）
LOOK = 5.0                # 隅の角度は、前後 5m 先の点との向きで測る（筆ポリゴンの細かいギザギザを拾わないため）
CORNER_MAX = 150.0        # これより鈍い角は隅にしない
CORNER_GAP = 10.0         # 隅どうしはこれより離す
ACUTE = 75.0


def to_local(geom):
    """経緯度 → その田の中心を原点にした平面（m）"""
    import numpy as np, shapely
    c = geom.centroid; k = math.cos(math.radians(c.y))
    return shapely.transform(geom, lambda xy: np.column_stack([(xy[:, 0] - c.x) * 111320 * k, (xy[:, 1] - c.y) * 110540]))


def passes(poly, theta):
    """向き theta（x 軸からの角度・度）に走るときの行程の数と合計の長さ"""
    import numpy as np, shapely
    from shapely import affinity
    r = affinity.rotate(poly, -theta, origin=(0, 0))
    x0, y0, x1, y1 = r.bounds
    n = max(1, int(math.ceil((y1 - y0) / W)))
    ys = y0 + (np.arange(n) + 0.5) * (y1 - y0) / n                  # 幅 W 以下の間隔で、端まで覆うように並べる
    lines = shapely.linestrings(np.stack([np.stack([np.full(n, x0 - 1), ys], 1), np.stack([np.full(n, x1 + 1), ys], 1)], 1))
    parts = shapely.get_parts(shapely.intersection(lines, r))
    ln = shapely.length(parts)
    ln = ln[ln >= MIN_PASS]
    return len(ln), float(ln.sum())


def corners(poly):
    """凸の隅の内角（度）の一覧"""
    import shapely
    from shapely.geometry import Polygon
    from shapely.geometry.polygon import orient
    out = []
    for part in getattr(poly, "geoms", [poly]):
        ring = orient(Polygon(part.exterior), 1.0).exterior          # 反時計回り
        L = ring.length
        if L < 4 * LOOK:
            continue
        pts = list(ring.coords)[:-1]
        s = [ring.project(shapely.Point(p)) for p in pts]
        cand = []
        for (x, y), si in zip(pts, s):
            b = ring.interpolate((si - LOOK) % L); f = ring.interpolate((si + LOOK) % L)
            ax, ay, bx, by = b.x - x, b.y - y, f.x - x, f.y - y
            na, nb = math.hypot(ax, ay), math.hypot(bx, by)
            if na < 1e-6 or nb < 1e-6:
                continue
            ang = math.degrees(math.acos(max(-1, min(1, (ax * bx + ay * by) / (na * nb)))))
            cross = bx * ay - by * ax                                  # 反時計回りの輪で、凸の隅は正
            if cross > 0 and ang < CORNER_MAX:
                cand.append((ang, si))
        taken = []
        for ang, si in sorted(cand):
            if all(min(abs(si - t), L - abs(si - t)) >= CORNER_GAP for _, t in taken):
                taken.append((ang, si))
        out += [a for a, _ in taken]
    return out


def n_back(a):
    return 90.0 / a if a <= 90 else (CORNER_MAX - a) / (CORNER_MAX - 90) if a < CORNER_MAX else 0.0


def a_man(a):
    return W * W / (4 * math.tan(math.radians(a) / 2)) if a < CORNER_MAX else 0.0


def metrics(geom):
    """1枚の田の [lbar, loss, perim, nacute, minang, dir]（計算できなければ None）"""
    g = to_local(geom).buffer(0)
    if g.is_empty or g.area < 50:
        return None
    A = g.area
    gs = g.simplify(1.0, preserve_topology=True)                       # 座標の丸め（約1m）のギザギザをならす
    P = gs.length
    # 作業の向きの候補: 田の辺（8m以上）と外接長方形の辺
    cand = []
    for geo in list(getattr(g.simplify(2.0), "geoms", [g.simplify(2.0)])) + [g.minimum_rotated_rectangle]:
        cs = list(geo.exterior.coords)
        for (x0, y0), (x1, y1) in zip(cs, cs[1:]):
            ln = math.hypot(x1 - x0, y1 - y0)
            if ln >= 8:
                cand.append((ln, math.degrees(math.atan2(y1 - y0, x1 - x0)) % 180))
    dirs = []
    for ln, th in sorted(cand, reverse=True):
        if all(min(abs(th - d), 180 - abs(th - d)) > 3 for d in dirs):
            dirs.append(th)
        if len(dirs) >= 8:
            break
    best = None
    for th in dirs or [0.0]:
        n, tot = passes(g, th)
        if n and (best is None or n < best[0] or (n == best[0] and tot > best[1])):
            best = (n, tot, th)
    if not best:
        return None
    n, tot, th = best
    cs = corners(gs)
    T = A / (W * V) + n * T_TURN + sum(T_BACK * n_back(a) + a_man(a) * C_MAN for a in cs)
    B = math.sqrt(A / 3)
    T0 = A / (W * V) + max(1, math.ceil(B / W)) * T_TURN + 4 * (T_BACK * n_back(90) + a_man(90) * C_MAN)
    acute = [a for a in cs if a < ACUTE]
    north = (90 - th) % 180                                            # x 軸（東）からの角 → 北から時計回り
    return [round(tot / n), round(100 * (T / T0 - 1)), round(100 * P / (8 * B)), len(acute), round(min(cs)) if cs else 90, round(north)]


def update_cell(cfg, data_dir, cell, log=None, deadline=None):
    from shapely.geometry import shape
    cdir = os.path.join(data_dir, "cells", cell["id"])
    parcels = read_json(os.path.join(cdir, "parcels.geojson"))
    if not parcels or not parcels.get("features"):
        return 0
    out = {}
    for f in parcels["features"]:
        try:
            m = metrics(shape(f["geometry"]))
        except Exception:
            m = None
        if m:
            out[f["properties"]["pid"]] = m
    write_json(os.path.join(cdir, "shape.json"), {"v": VERSION, "W": W, "p": out})
    return 1


def need(data_dir, cell):
    m = read_json(os.path.join(data_dir, "cells", cell["id"], "shape.json")) or {}
    return m.get("v") != VERSION
