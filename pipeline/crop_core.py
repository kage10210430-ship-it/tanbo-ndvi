"""crop_core: rice / 麦 / 休 / 他 per paddy field and year (Fukui, 2022-).  numpy only, no training at runtime.

Final design ("F") = design C (evidence scores -> small logistic model per data regime -> rotation/neighbour context)
plus evidence fixes (fixes 1-2: DECISION.md; fix 4 and the radar re-scaling: final_new/RECAL.md):
  * late vigour : a field that was flooded (radar / trend R / open water) and greened up (NDVI 0.5) by 8/3 may show its
                  canopy in August instead of July (late transplanting, 乾田直播).
  * flooded weeds: "green all spring (never bare)" does not count against rice when radar shows flooding
                  (S1 series or trend R <= -20..-23 dB), except in tiny fields (< 3 inner pixels) whose radar is mixed.
  * late puddling: open water or two flooded radar scenes in June in a field that was dry in spring = transplanting in
                  June (late rice, rice after barley); "green in late May" / "never bare" are then weeds or barley before
                  puddling and do not count against rice (>= 3 inner pixels).
F-4 (final_v4/FIX.md; starts from F-2 robust, fixes the F-3 blind-verifier issues 1-11):
  1 flood period: best 30-day window 4/20-7/20, k-th lowest VH (k = max(2, n/4)) -> late floods are not diluted
  2 late-canopy path gated by a saturating water ramp (no flood x flood product)
  3 'no_water' evidence: dense radar series (>= 12 scenes), >= 3 px, no water of any kind, never bare in May
  4 harvest drop may be followed by regrowth (next obs >= 0.2 below the peak)
  5 with water: late canopy up to 9/20 (2nd highest NDVI 8/01-9/20), green-up by 8/8; weak canopy 0.50->0.70 with a
    rice-time harvest (F-3); harvest window 7/25-10/25
  6 late-puddling dry test tolerates one (<= 20 %) wet early scene
  7 barley shape x (1 - June water x canopy) (麦後稲); clear open water NDVI <= 0 at any size; after a green spring,
    low optical NDVI alone is bare soil (麦後大豆) and is not water
  8 per-field orbit bias: an acquisition group >= 7 dB darker than the brightest (scene pairs <= 6 days) is raised
  10 barley shape from the 2nd highest NDVI 4/01-5/20 ramp 0.55 -> 0.72 (spring weeds, one odd obs)
  11 'noharv_green': green in October without a rice harvest
  (F-3 ports: 2-point haze dip removal, the harvest-conditional weak canopy.  Not ported: R only as fallback, bounded
  context, cell-level orbit offsets, flood confirmation, stricter optical late puddling.)
Radar: parcel-median VH (radar.json 2022-2024, ndvi.json rp 2025-), unquantised; thresholds are 0.5 dB lower than for
the old quantised parcel-mean data (VH_FLOOD, Q20_RAMP; RECAL.md).

Stage 1 (per cell, independent):  stage1_cell(cell, J, model) -> list of field-year records (own-year evidence only).
Stage 2 (across cells):           stage2(records, model, season_year) -> results; `records` = the cells to label plus
                                  (optionally) their neighbour cells' stage-1 records; restrict output with `only_cells`.
One call:                         label_cell(cell, J, model, neighbour_records=None, season_year=None).

Production robustness (robustness.patch): unreadable files are reported, not fatal (load_cell '_errors'); per-year files
count only for the years listed in their "years"; every field-year carries `cov` (which needed windows its files hold)
and `hz` (how far its data reach): incomplete inputs are flagged (`incomplete`, reason "（データ不足）"), a year whose
data end before 10/20 stays provisional whatever the calendar says, and a running year with data before 9/01 is all
provisional and kept out of the rotation / neighbour context; field-years without any own-year data get no label.

J = dict of loaded JSON of one cell: parcels, trend, ndvi, hist, spring and, when present, radar, pre, post, hist_ls
(missing file -> None / absent).  New files are used automatically; the data regime (radar series yes/no, autumn optical
yes/no) selects the stage-1 weights, so years move to the richer model as soon as their files exist.
"""
import json, math, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, 'crop_model.json')
NAN = float('nan')
VERSION = 'F-4'

# ------------------------------------------------------------------ feature constants
CLEAR_MIN = 80                          # clearPct (%) for optical and radar obs
LS_FIT_DAYS, LS_FIT_TOL = 20, 0.15      # Landsat obs dropped when >= 0.15 off the S2 line (viewer lsFits)
DIP, DIP_DAYS, DIP_PASSES = 0.10, 12, 2  # V-dip (haze) removal
WATER_NDVI, WATER_KEEP = 0.05, (4, 6)   # open water in Apr-Jun is never removed as a dip
VH_FLOOD = -21.25                       # S1 VH <= -21.25 dB on the parcel-median radar (= old -21 dB on the quantised
                                        # parcel mean; same-date paired shift, RECAL.md)
FLOOD_WIN = (420, 710)                  # radar flood window (MMDD)
# ---- F-4 (final_v4/FIX.md) ----
DIP2_SPAN, DIP2_RETURN, DIP2_MD, DIP2_TOP = 25, 0.12, (701, 930), 0.60   # 2-point haze dips in Jul-Sep (from F-3):
                                        # both obs >= DIP below both neighbours, which are canopy (>= 0.60), <= 25 days
                                        # apart and within 0.12 of each other (a harvest never returns to the canopy)
VH_GROUP = 12                           # acquisition group = day number mod 12 (satellite x relative orbit)
VH_GAP_DAYS, VH_GAP_MIN, VH_GAP_FIX = 6, 4, 7.0   # per-FIELD orbit bias: groups compared on scene pairs <= 2 days
                                        # apart (>= 4 pairs, >= 80 % the same sign); when one group is >= 6 dB darker than the
                                        # brightest, its scenes are raised to that level (shadow / layover, not water)
W30_WIN, W30_DAYS = (420, 720), 30      # flood period: best 30-day window 4/20-7/20; statistic = k-th lowest VH in
                                        # it, k = max(2, ceil(n/4)) (a sustained low, not one odd scene)
W30_RAMP = (-20.0, -23.0)
EARLY_WET_SHARE = 0.2                   # late-puddling dry test: <= max(1, 20 %) of the 4/20-5/25 scenes <= -21 dB
HARV_WIN = (217, 283)                   # rice harvest drop 8/05-10/10 ...
HARV_WIN_WET = (206, 298)               # ... 7/25-10/25 when the field shows water (ハナエチゼン early Aug; late
                                        # varieties / rice after barley mid-late Oct)
HARV_BACK = 0.2                         # a regrowth after the drop is fine while it stays >= 0.2 below the peak
LATE_VIG_MD = (801, 920)                # late canopy window (2nd highest NDVI) when the field shows water
VIG_HARV_RAMP = (0.50, 0.70)            # canopy (2nd highest NDVI 7/01-8/31) of a field with water AND a rice-time harvest
                                        # (small / late / hot-summer rice; F-3): 0.50 -> 0.70
LATE_GREEN_DOY = (215, 220)             # green-up (NDVI 0.5) by 8/3; by 8/8 with water (transplant late June)
NOWATER_MIN_N = 12                      # 'no water' evidence needs a dense radar series (>= 12 scenes 4/20-7/10)
WATER_GATE = (0.2, 0.5)                 # saturating water gate for the late-canopy path (no flood x flood product)
LP_WATER_RAMP = (0.02, -0.02)           # clear open water (NDVI <= 0; bare soil after barley reaches 0.03-0.10)
BARLEY_RAMP = (0.55, 0.72)              # barley shape: 2nd highest NDVI 4/01-5/20 (one odd obs / spring weeds ~0.55)
OFF = set(x for x in os.environ.get('F4_OFF', '').split(',') if x)   # ablation switches (development only)
LATE_PUDDLE = True                      # evidence fix 4 (RECAL.md)
LP_VH_RAMP = (-22.0, -24.5)             # late puddling: 2nd deepest VH 6/01-7/10 (dB)
LP_DRY_EARLY = -19.0                    # ... only when VH 4/20-5/25 was dry: median >= -19 dB and no scene <= -21 dB
FRAC_RAMP = (0.10, 0.50)                # flood_s1: share of flooded scenes 0.10 -> 0.50
Q20_RAMP = (-20.0, -23.0)               # flood_s1: 20th percentile VH (dB); old quantised-mean data: -19.5 -> -22.5
# windows the classifier needs per year (MMDD) and the per-year file that holds them when ndvi.json does not
# (trend.py hist_years: a year goes to the file when the window starts before ndvi.json's first date)
COVER = [('pre', 'pre', 301, 414), ('spring', 'spring', 415, 610), ('hist', 'summer', 510, 930),
         ('post', 'autumn', 1001, 1130), ('radar', 'radar', 415, 810)]
SEASON_CLOSED_MD = 1020                 # a year's optical data reaching 10/20 = season over (same rule as regime P1)
SEASON_LATE_MD = 901                    # before the data reach 9/01 every label of the running year is provisional
FILES = [('parcels', 'parcels.geojson'), ('trend', 'trend.json'), ('ndvi', 'ndvi.json'), ('hist', 'hist.json'),
         ('spring', 'spring.json'), ('pre', 'pre.json'), ('post', 'post.json'), ('radar', 'radar.json'),
         ('hist_ls', 'hist_ls.json')]
TREND_KEYS = ['A', 'J', 'R', 'ne', 'me', 'nl', 'ml']
TREND_SCALE = [1000, 1000, 10, 1, 1000, 1, 1000]
OPT_KEYS = ['n_opt', 'n_pre', 'n_post', 'last_md', 'mar_o', 'apr_o', 'n_may', 'may_o', 'emay_o', 'bare_o', 'n_bare_o',
            'lmay_o', 'jmax2', 'smin', 'smin_doy', 'n_sw', 'n_low', 'low_first', 'low_last', 'wmin', 'bmax', 'bmin',
            'bmin_doy', 'jl_o', 'ju_o', 'n_j', 'jmax', 'aug_o', 'green_doy', 'peak', 'peak_doy', 'harv_doy', 'harv_min',
            'sep2_o', 'oct_o', 'nov_o', 'au_min', 'sum_min', 'n_dip', 'harv2_doy', 'lenv', 'bmax2', 'env2']
VH_KEYS = ['vh_n', 'vh_q20', 'vh_q50', 'vh_frac', 'vh_n21', 'vh_early', 'vh_late', 'vh_can', 'vh_rise', 'vh_min_doy',
           'vh_kind', 'vh_n_all', 'vh_jmin2', 'vh_emin', 'vh_w30', 'vh_w30_doy', 'vh_ne', 'vh_ne21', 'vh_gap']
NUM_KEYS = ['area_m2', 'full', 'lon', 'lat', 'cov', 'hz'] + TREND_KEYS + OPT_KEYS + VH_KEYS

STATES = ['稲', '麦', '他']
SZ_BINS = [3, 5, 10, 20]                # size classes <3, 3-5, 5-10, 10-20, >=20 inner 10 m pixels
FALLOW_VIG = 0.6                        # July vigour score of a green, unharvested non-rice field -> 休
LAT0 = 36.0                             # fixed projection latitude for the 150 m neighbour search


# ================================================================== helpers
def load_model(path=MODEL_PATH):
    with open(path, encoding='utf-8') as fh:
        return json.load(fh)


def load_cell(cdir):
    """Missing file -> None.  An unreadable / half-written file (the pipeline writes JSON in place) is also None, and
    its name is listed in J['_errors'] so the caller can report it; one bad file must not stop the whole run."""
    J = {'_errors': []}
    for key, fn in FILES:
        try:
            with open(os.path.join(cdir, fn), encoding='utf-8') as fh:
                J[key] = json.load(fh)
        except FileNotFoundError:
            J[key] = None
        except (OSError, ValueError) as e:              # JSONDecodeError is a ValueError
            J[key] = None; J['_errors'].append(f'{fn}: {type(e).__name__}')
    return J


def date_days(dates):
    return np.array(dates or [], dtype='datetime64[D]').astype(np.int64)


def md_doy(t):
    d = np.asarray(t).astype('datetime64[D]')
    m = d.astype('datetime64[M]'); y = d.astype('datetime64[Y]')
    return ((m.astype(np.int64) % 12 + 1) * 100 + (d - m).astype(np.int64) + 1,
            (d - y).astype(np.int64) + 1, y.astype(np.int64) + 1970)


def day0(year, month=1, day=1):
    return int(np.datetime64(f'{year:04d}-{month:02d}-{day:02d}', 'D').astype(np.int64))


def triples(a):
    a = np.asarray(a, dtype=np.float64).reshape(-1, 3)
    return a[:, 0].astype(np.int64), a[:, 1], a[:, 2]


def _med(v):
    return float(np.median(v)) if v.size else NAN


def _min(v):
    return float(v.min()) if v.size else NAN


def _max(v):
    return float(v.max()) if v.size else NAN


def inwin(md, a, b):
    return (md >= a) & (md <= b)


_A, _E2 = 6378137.0, 6.69437999014e-3


def geom_area_centroid(geom):
    """Area (m^2) and centroid (lon, lat) of a GeoJSON (Multi)Polygon (local ellipsoidal projection)."""
    polys = [geom['coordinates']] if geom['type'] == 'Polygon' else geom['coordinates']
    ext = np.asarray(polys[0][0], dtype=np.float64)
    lon0, lat0 = ext[:, 0].mean(), ext[:, 1].mean()
    phi = math.radians(lat0); s2 = math.sin(phi) ** 2
    N = _A / math.sqrt(1 - _E2 * s2); M = _A * (1 - _E2) / (1 - _E2 * s2) ** 1.5
    kx, ky = math.radians(1) * N * math.cos(phi), math.radians(1) * M
    area = cx = cy = 0.0
    for poly in polys:
        for k, ring in enumerate(poly):
            r = np.asarray(ring, dtype=np.float64)
            x = (r[:, 0] - lon0) * kx; y = (r[:, 1] - lat0) * ky
            x1, y1 = np.roll(x, -1), np.roll(y, -1)
            cr = x * y1 - x1 * y
            a = cr.sum() / 2
            if a == 0:
                continue
            sx = ((x + x1) * cr).sum() / (6 * a); sy = ((y + y1) * cr).sum() / (6 * a)
            a = abs(a) * (1 if k == 0 else -1)
            area += a; cx += a * sx; cy += a * sy
    if area <= 0:
        return 0.0, lon0, lat0
    return area, lon0 + cx / area / kx, lat0 + cy / area / ky


# ================================================================== series cleaning
def remove_dips(t, v, md):
    keep = np.ones(len(t), bool)
    mon = md // 100
    protect = (v < WATER_NDVI) & (mon >= WATER_KEEP[0]) & (mon <= WATER_KEEP[1])
    for _ in range(DIP_PASSES):
        idx = np.flatnonzero(keep)
        if len(idx) < 3:
            break
        tt, vv = t[idx], v[idx]
        d = np.zeros(len(idx), bool)
        d[1:-1] = ((tt[1:-1] - tt[:-2] <= DIP_DAYS) & (tt[2:] - tt[1:-1] <= DIP_DAYS) &
                   (vv[:-2] - vv[1:-1] >= DIP - 1e-9) & (vv[2:] - vv[1:-1] >= DIP - 1e-9))
        d &= ~protect[idx]
        if not d.any():
            break
        keep[idx[d]] = False
    # two consecutive hazy obs (F-3 fix 2, kept in F-4): a 2-obs dip in Jul-Sep that returns to the canopy level
    idx = np.flatnonzero(keep)
    if len(idx) >= 4 and 'dip2' not in OFF:
        tt, vv, mm = t[idx], v[idx], md[idx]
        a, b, c, e = vv[:-3], vv[1:-2], vv[2:-1], vv[3:]
        lo = np.fmin(a, e) - DIP + 1e-9
        d2 = ((b <= lo) & (c <= lo) & (np.abs(a - e) <= DIP2_RETURN + 1e-9) & (tt[3:] - tt[:-3] <= DIP2_SPAN) &
              (np.fmin(a, e) >= DIP2_TOP) & (mm[1:-2] >= DIP2_MD[0]) & (mm[2:-1] <= DIP2_MD[1]))
        if d2.any():
            j = np.flatnonzero(d2)
            keep[idx[np.r_[j + 1, j + 2]]] = False
    return keep


def merge_optical(pid, s2src, lssrc):
    """Clear obs of all S2 sources (priority order) + Landsat on dates without S2 (checked against the S2 line),
    haze V-dips removed."""
    ts, vs, ls = [], [], []
    for srcs, isl in ((s2src, False), (lssrc, True)):
        for t0, p in srcs:
            a = p.get(pid) if p else None
            if not a:
                continue
            idx, val, pct = triples(a)
            ok = (pct >= CLEAR_MIN) & (idx < len(t0))
            ts.append(t0[idx[ok]]); vs.append(val[ok] / 1000); ls.append(np.full(int(ok.sum()), isl))
    if not ts:
        return None
    t = np.concatenate(ts); v = np.concatenate(vs); l = np.concatenate(ls)
    if not len(t):
        return None
    t, first = np.unique(t, return_index=True)
    v, l = v[first], l[first]
    md, doy, yr = md_doy(t)
    if l.any() and (~l).sum() >= 2:
        st, sv = t[~l], v[~l]
        li = np.flatnonzero(l)
        pos = np.searchsorted(st, t[li], 'left')
        ok = (pos > 0) & (pos < len(st))
        bad = np.zeros(len(li), bool)
        if ok.any():
            p0, p1 = pos[ok] - 1, pos[ok]
            tl = t[li[ok]]
            near = (tl - st[p0] <= LS_FIT_DAYS) & (st[p1] - tl <= LS_FIT_DAYS)
            line = sv[p0] + (sv[p1] - sv[p0]) * (tl - st[p0]) / (st[p1] - st[p0])
            bad[ok] = near & (np.abs(v[li[ok]] - line) >= LS_FIT_TOL - 1e-9)
        keep = np.ones(len(t), bool); keep[li[bad]] = False
        t, v, l, md, doy, yr = t[keep], v[keep], l[keep], md[keep], doy[keep], yr[keep]
    keep = remove_dips(t, v, md)
    dips = {}
    for y in yr[~keep]:
        dips[int(y)] = dips.get(int(y), 0) + 1
    return dict(t=t[keep], v=v[keep], md=md[keep], doy=doy[keep], yr=yr[keep], dips=dips)


def merge_radar(pid, rsrc):
    """rsrc: list of (days, rp dict, kind) in priority order (kind 1 = parcel median, 0 = quantised mean)."""
    ts, vs, ks = [], [], []
    for t0, p, kind in rsrc:
        a = p.get(pid) if p else None
        if not a:
            continue
        idx, val, pct = triples(a)
        ok = (pct >= CLEAR_MIN) & (idx < len(t0))
        ts.append(t0[idx[ok]]); vs.append(val[ok] / 1000); ks.append(np.full(int(ok.sum()), kind))
    if not ts:
        return None
    t = np.concatenate(ts); v = np.concatenate(vs); k = np.concatenate(ks)
    if not len(t):
        return None
    t, first = np.unique(t, return_index=True)
    md, doy, yr = md_doy(t)
    return dict(t=t, v=v[first], k=k[first], md=md, doy=doy, yr=yr, g=t % VH_GROUP)


def orbit_fix(t, v, g):
    """Per-field orbit bias (F-4 fix 8).  Offsets of the acquisition groups from scene pairs of different groups <= 2
    days apart (median differences, least squares, brightest group = 0).  A group >= 6 dB darker than the brightest is a
    viewing-geometry artefact (shadow / mixed pixel), not water: its scenes are raised by the offset.  Returns (v', gap)."""
    gs = np.unique(g)
    if len(gs) < 2 or len(v) < 6:
        return v, 0.0
    D = {}
    for lag in (1, 2, 3, 4):
        if len(t) <= lag:
            break
        i = np.arange(len(t) - lag); j = i + lag
        ok = (t[j] - t[i] <= VH_GAP_DAYS) & (g[i] != g[j])
        for a, b, dv in zip(g[i[ok]], g[j[ok]], v[j[ok]] - v[i[ok]]):
            D.setdefault((int(a), int(b)), []).append(float(dv))
    eqs = [(a, b, float(np.median(d)), len(d)) for (a, b), d in D.items()
           if len(d) >= VH_GAP_MIN and max(np.mean(np.asarray(d) > 0), np.mean(np.asarray(d) < 0)) >= 0.8]
    if not eqs:
        return v, 0.0
    us = sorted({a for a, _, _, _ in eqs} | {b for _, b, _, _ in eqs}); ix = {x: n for n, x in enumerate(us)}
    A = np.zeros((len(eqs) + 1, len(us))); rhs = np.zeros(len(eqs) + 1)
    for n, (a, b, m, c) in enumerate(eqs):
        s_ = math.sqrt(c); A[n, ix[b]] = s_; A[n, ix[a]] = -s_; rhs[n] = s_ * m
    A[-1] = 1e-3
    o = np.linalg.lstsq(A, rhs, rcond=None)[0]
    o = o - o.max()                                            # brightest group = 0, others <= 0
    gap = float(-o.min())
    if gap < VH_GAP_FIX or 'orbit' in OFF:
        return v, gap
    corr = np.zeros(len(v))
    for x, oo in zip(us, o):
        if -oo >= VH_GAP_FIX:
            corr[g == x] = -oo
    return v + corr, gap


# ================================================================== per-year measures
def optical_year(s, year):
    f = dict.fromkeys(OPT_KEYS, NAN)
    f.update(n_opt=0, n_pre=0, n_post=0, n_may=0, n_sw=0, n_low=0, n_j=0, n_dip=0, n_bare_o=0)
    if s is None:
        return f
    f['n_dip'] = s['dips'].get(year, 0)
    i0, i1 = np.searchsorted(s['t'], [day0(year), day0(year + 1)])
    if i1 <= i0:
        return f
    v, md, doy = s['v'][i0:i1], s['md'][i0:i1], s['doy'][i0:i1]
    f['n_opt'] = int(inwin(md, 415, 930).sum())
    f['n_pre'] = int(inwin(md, 301, 414).sum())
    f['n_post'] = int(inwin(md, 1001, 1130).sum())
    w = md <= 1130
    f['last_md'] = int(md[w][-1]) if w.any() else NAN
    f['mar_o'] = _med(v[inwin(md, 301, 331)])
    f['apr_o'] = _med(v[inwin(md, 401, 430)])
    w = inwin(md, 501, 525); f['n_may'] = int(w.sum()); f['may_o'] = _med(v[w])
    f['emay_o'] = _med(v[inwin(md, 425, 515)])
    w = inwin(md, 425, 531); f['n_bare_o'] = int(w.sum()); f['bare_o'] = _med(v[w])
    f['lmay_o'] = _med(v[inwin(md, 516, 531)])
    f['jmax2'] = _max(v[inwin(md, 701, 810)])
    w = inwin(md, 415, 630)
    f['n_sw'] = int(w.sum())
    imin = None
    if w.any():
        j = np.flatnonzero(w); imin = j[np.argmin(v[j])]
        f['smin'] = float(v[imin]); f['smin_doy'] = int(doy[imin])
        lo = j[v[j] < 0.3]
        f['n_low'] = int(len(lo))
        if len(lo):
            f['low_first'] = int(doy[lo[0]]); f['low_last'] = int(doy[lo[-1]])
    f['wmin'] = _min(v[inwin(md, 415, 620)])
    f['bmax'] = _max(v[inwin(md, 401, 520)])
    w = inwin(md, 521, 710)
    if w.any():
        j = np.flatnonzero(w); k = j[np.argmin(v[j])]
        f['bmin'] = float(v[k]); f['bmin_doy'] = int(doy[k])
    f['jl_o'] = _med(v[inwin(md, 616, 710)])
    f['ju_o'] = _med(v[inwin(md, 711, 731)])
    w = inwin(md, 701, 831); f['n_j'] = int(w.sum()); f['jmax'] = _max(v[w])
    f['aug_o'] = _med(v[inwin(md, 801, 831)])
    f['sum_min'] = _min(v[inwin(md, 611, 720)])
    if imin is not None:
        j = np.flatnonzero((np.arange(len(v)) > imin) & (v >= 0.5) & (md <= 1015))
        if len(j):
            j = j[0]; a, b = v[j - 1], v[j]
            f['green_doy'] = float(doy[j - 1] + (0.5 - a) / (b - a) * (doy[j] - doy[j - 1])) if b > a else float(doy[j])
    w = inwin(md, 601, 1031)
    if w.any():
        j = np.flatnonzero(w); ip = j[np.argmax(v[j])]
        pk = float(v[ip]); f['peak'] = pk; f['peak_doy'] = int(doy[ip])
        thr = max(0.45, pk - 0.3)
        low = v < thr
        sus = low & np.append(low[1:], True)          # stays low at the next obs (or is the last obs)
        h = np.flatnonzero((np.arange(len(v)) > ip) & sus & (md <= 1130))
        if len(h):
            f['harv_doy'] = int(doy[h[0]])
        # F-4 fix 4: a regrowth (ratoon, weeds) after the drop is fine while the next obs stays >= 0.2 below the peak
        sus2 = low & np.append(v[1:] < pk - HARV_BACK, True)
        h2 = np.flatnonzero((np.arange(len(v)) > ip) & sus2 & (md <= 1130))
        if len(h2):
            f['harv2_doy'] = int(doy[h2[0]])
        f['harv_min'] = _min(v[(np.arange(len(v)) > ip) & (md <= 1130)])
    x = np.sort(v[inwin(md, 701, 831)])[::-1]                     # summer upper envelope (haze-cleaned; F-3)
    if len(x):
        f['env2'] = float(x[1]) if len(x) >= 2 else float(x[0]) - 0.05
    x = np.sort(v[inwin(md, *LATE_VIG_MD)])[::-1]
    if len(x):
        f['lenv'] = float(x[1]) if len(x) >= 2 else float(x[0]) - 0.05
    x = np.sort(v[inwin(md, 401, 520)])[::-1]
    if len(x):
        f['bmax2'] = float(x[1]) if len(x) >= 2 else float(x[0]) - 0.05
    f['sep2_o'] = _med(v[inwin(md, 911, 930)])
    f['oct_o'] = _med(v[inwin(md, 1001, 1031)])
    f['nov_o'] = _med(v[inwin(md, 1101, 1130)])
    f['au_min'] = _min(v[inwin(md, 810, 1015)])
    return f


def radar_year(r, year):
    f = dict.fromkeys(VH_KEYS, NAN)
    f.update(vh_n=0, vh_n21=0, vh_n_all=0)
    if r is None:
        return f
    i0, i1 = np.searchsorted(r['t'], [day0(year), day0(year + 1)])
    if i1 <= i0:
        return f
    v, md, doy, k = r['v'][i0:i1], r['md'][i0:i1], r['doy'][i0:i1], r['k'][i0:i1]
    tt = r['t'][i0:i1]
    w = inwin(md, 415, 810)
    if w.sum() >= 6:
        vw, gap = orbit_fix(tt[w], v[w], r['g'][i0:i1][w])
        v = v.copy(); v[w] = vw; f['vh_gap'] = gap
    f['vh_n_all'] = int(len(v)); f['vh_kind'] = float(k.mean())
    # F-4 fix 1: the field's own flood period = best 30-day window: k-th lowest VH, k = max(2, ceil(n/4))
    w = np.flatnonzero(inwin(md, *W30_WIN))
    if len(w) >= 2:
        tw, vw = tt[w], v[w]
        end = np.searchsorted(tw, tw + W30_DAYS, 'left')          # window [t_a, t_a + 30)
        cnt = end - np.arange(len(w))
        L = int(cnt.max())
        M = np.full((len(w), L), np.inf)
        ii = np.arange(len(w))[:, None] + np.arange(L)[None, :]
        ok = np.arange(L)[None, :] < cnt[:, None]
        M[ok] = vw[ii[ok]]
        M.sort(1)
        kk = np.maximum(2, -(-cnt // 4))
        stat = np.where(cnt >= 2, M[np.arange(len(w)), np.minimum(kk, L) - 1], np.inf)
        a = int(np.argmin(stat))
        if np.isfinite(stat[a]):
            f['vh_w30'] = float(stat[a]); f['vh_w30_doy'] = int(doy[w[a]])
    e = v[inwin(md, 420, 525)]
    f['vh_ne'] = int(len(e)); f['vh_ne21'] = int((e <= VH_FLOOD).sum())
    w = inwin(md, *FLOOD_WIN)
    n = int(w.sum()); f['vh_n'] = n
    if n:
        x = v[w]
        f['vh_q20'] = float(np.quantile(x, 0.2)); f['vh_q50'] = float(np.median(x))
        f['vh_n21'] = int((x <= VH_FLOOD).sum()); f['vh_frac'] = f['vh_n21'] / n
        j = np.flatnonzero(w); o = j[np.argsort(v[j])]
        f['vh_min_doy'] = int(doy[o[min(1, len(o) - 1)]])
    f['vh_early'] = _med(v[inwin(md, 420, 525)])
    f['vh_late'] = _med(v[inwin(md, 526, 630)])
    f['vh_can'] = _med(v[inwin(md, 711, 810)])
    x = np.sort(v[inwin(md, 601, 710)])
    f['vh_jmin2'] = float(x[1]) if len(x) >= 2 else NAN            # 2nd lowest: one odd scene is not a flood
    f['vh_emin'] = _min(v[inwin(md, 420, 525)])
    if n:
        f['vh_rise'] = f['vh_can'] - f['vh_q20']
    return f


def _dates(j, key='dates'):
    return date_days(j.get(key)) if j else np.zeros(0, np.int64)


def _done_years(j):
    """Per-year files: only the years listed in "years" are complete (trend.py saves after every chunk of dates and
    adds the year at the end, so a killed run leaves a partial year).  Files without "years" are used as they are."""
    if not j:
        return None
    ys = j.get('years')
    return None if ys is None else {int(y) for y in ys}


def _only_years(t0, p, done):
    """Drop the observations of years not in `done` (dates -> day numbers t0; p = {pid: [i, v, pct, ...]})."""
    if done is None or not p or not len(t0):
        return t0, p
    yr = t0.astype('datetime64[D]').astype('datetime64[Y]').astype(np.int64) + 1970
    ok = np.isin(yr, sorted(done))
    if ok.all():
        return t0, p
    good = ok                                                  # rare case (a partial year): filter the triples
    q = {}
    for pid, arr in p.items():
        x = np.asarray(arr, dtype=np.int64).reshape(-1, 3)
        x = x[(x[:, 0] < len(t0)) & good[np.clip(x[:, 0], 0, len(t0) - 1)]]
        if len(x):
            q[pid] = x.ravel().tolist()
    return t0, q


def coverage(J, year):
    """Which of the needed windows of `year` the cell's files hold (dict name -> bool), and whether the year's optical
    series reaches: hz = MMDD of ndvi.json's last date in `year` (1231 when it goes beyond, 0 when it ends before).
    ndvi.json holds a window when its first date is on/before the window start
    (it holds everything after that up to its last date); otherwise the per-year file must list the year as done."""
    nd = J.get('ndvi') or {}
    ds = nd.get('dates') or []
    first, last = (ds[0], ds[-1]) if ds else ('9999-99-99', '0000-00-00')
    cov = {}
    for key, name, a, b in COVER:
        s0 = f'{year:04d}-{a // 100:02d}-{a % 100:02d}'; s1 = f'{year:04d}-{b // 100:02d}-{b % 100:02d}'
        if first <= s0:
            cov[name] = last >= s1 or last[:4] == str(year)       # the running year: covered up to its last date
        else:
            d = _done_years(J.get(key))
            cov[name] = bool(J.get(key)) and (d is None or year in d)
    hz = 1231 if last[:4] > str(year) else (int(last[5:7] + last[8:10]) if last[:4] == str(year) else 0)
    return cov, hz


def cell_features(cell, J, years=None):
    """One dict per field-year of the cell (the years of the pid in trend.json, or the cell's trend "years" for a pid
    not yet in trend.json, e.g. a new parcel; restricted to `years` when given)."""
    parcels, trend = J.get('parcels'), J.get('trend')
    if not parcels or not trend:
        return []
    nd = J.get('ndvi') or {}
    s2 = [(_dates(nd), nd.get('p'))]
    s2 += [_only_years(_dates(J[k]), J[k].get('p'), _done_years(J[k])) for k in ('hist', 'spring', 'pre', 'post') if J.get(k)]
    ls = [(_dates(nd, 'ldates'), nd.get('lp'))]
    if J.get('hist_ls'):
        ls.append(_only_years(_dates(J['hist_ls'], 'ldates'), J['hist_ls'].get('lp'), _done_years(J['hist_ls'])))
    rs = []
    if J.get('radar'):
        rj = J['radar']
        t0, p = _only_years(_dates(rj, 'rdates'), rj.get('rp'), _done_years(rj))
        rs.append((t0, p, 1 if 'med' in str(rj.get('rmask', rj.get('mask', 'med'))) else 0))
    rs.append((_dates(nd, 'rdates'), nd.get('rp'), 1 if 'med' in str(nd.get('rmask', '')) else 0))
    tp = trend.get('p', {})
    cell_years = [str(y) for y in (trend.get('years') or []) if isinstance(y, int)]
    covs = {}
    rows = []
    for feat in parcels['features']:
        pr = feat['properties']; pid = str(pr['pid'])
        yrs = tp.get(pid) or {y: None for y in cell_years}          # new parcel: trend values missing (NaN)
        if not yrs:
            continue
        try:
            area, lon, lat = geom_area_centroid(feat['geometry'])
        except (TypeError, KeyError, IndexError, ValueError):       # null / empty geometry: no area, no neighbours
            area, lon, lat = NAN, NAN, NAN
        s = merge_optical(pid, s2, ls); r = merge_radar(pid, rs)
        full = pr.get('full')
        for ys in sorted(yrs):
            y = int(ys)
            if years and y not in years:
                continue
            if y not in covs:
                cv, hz = coverage(J, y)
                covs[y] = (int(sum(1 << i for i, (_, n, _, _) in enumerate(COVER) if cv[n])), hz)
            row = dict(cell=cell, pid=pid, year=y, lon=round(lon, 6), lat=round(lat, 6), area_m2=area,
                       full=NAN if full is None else float(full), cov=covs[y][0], hz=covs[y][1])
            for k, sc, x in zip(TREND_KEYS, TREND_SCALE, yrs[ys] or [None] * 7):
                row[k] = NAN if x is None else x / sc
            row.update(optical_year(s, y)); row.update(radar_year(r, y))
            rows.append(row)
    return rows


# ================================================================== evidence scores
def ramp(x, a, b):
    """0 at a, 1 at b, linear in between (a > b: falling ramp); NaN stays NaN."""
    return np.clip((np.asarray(x, float) - a) / (b - a), 0, 1)


def _nz(x, v):
    x = np.asarray(x, float)
    return np.where(np.isnan(x), v, x)


def _first(*cols):
    out = np.asarray(cols[0], float).copy()
    for c in cols[1:]:
        out = np.where(np.isnan(out), np.asarray(c, float), out)
    return out


EVIDENCE = ['flood_s1', 'flood_R', 'flood_opt', 'bare', 'vigour', 'bare_x_vig', 'flood_x_vig', 'harvest',
            'late_green', 'early_green', 'barley', 'grass', 'small', 'floodR_small', 'flood_s1_small', 'no_water',
            'noharv_green']
EVIDENCE_JA = {'flood_s1': '湛水(レーダー)', 'flood_R': '湛水(R)', 'flood_opt': '水面(NDVI)', 'bare': '5月に裸地・水面',
               'vigour': '夏の生育', 'bare_x_vig': '代かき→生育', 'flood_x_vig': '湛水→生育', 'harvest': '刈取り',
               'late_green': '9月下旬も緑', 'early_green': '5月下旬に緑', 'barley': '麦(4月緑→6月刈取り)',
               'grass': '春から緑(耕起なし)', 'small': '小区画', 'floodR_small': '小区画の湛水(R)',
               'flood_s1_small': '小区画の湛水(レーダー)', 'no_water': '水の証拠なし', 'noharv_green': '10月も緑・刈取りなし'}


def columns(rows):
    """list of row dicts -> dict of float arrays (numeric keys) + object arrays (cell, pid) + int year."""
    F = {k: np.array([r[k] for r in rows], dtype=float) for k in NUM_KEYS}
    F['year'] = np.array([r['year'] for r in rows], dtype=int)
    F['pid'] = np.array([r['pid'] for r in rows], dtype=object)
    F['cell'] = np.array([r['cell'] for r in rows], dtype=object)
    return F


def evidence(F):
    """Evidence scores 0..1 (dict of arrays).  Breakpoints: agronomic calendar + cluster medians (DECISION.md, RECAL.md,
    final_v4/FIX.md).  Extra non-model arrays: late_puddle, water."""
    S = {}
    has_vh = F['vh_n'] >= 4
    s1 = np.fmax(ramp(F['vh_frac'], *FRAC_RAMP), ramp(F['vh_q20'], *Q20_RAMP))
    if 'w30' not in OFF:                   # F-4 fix 1: the field's own flood period (best 30-day window)
        s1 = np.fmax(s1, _nz(ramp(F['vh_w30'], *W30_RAMP), 0))
    S['flood_s1'] = np.where(has_vh, _nz(s1, 0), 0.0)
    S['flood_R'] = _nz(ramp(F['R'], -17.5, -22.0), 0)
    S['flood_opt'] = _nz(ramp(F['wmin'], 0.10, 0.0), 0)
    bo = _first(F['bare_o'], F['emay_o'], F['may_o'], F['A'])
    S['bare'] = _nz(ramp(bo, 0.45, 0.22), 0.5)
    full = _nz(F['full'], 1)
    # late puddling: open water (NDVI <= 0.1 after 6/1) or two flooded radar scenes 6/01-7/10 in a field that was dry in
    # 4/20-5/25 = transplanting in June; then "green in late May" / "never bare" / late green are weeds / barley before
    june = _nz(F['smin_doy'], 0) >= 152
    lp_opt = np.where(june, _nz(ramp(F['smin'], 0.15, 0.05), 0), 0.0)
    if 'lpdry' in OFF:
        dry = (_nz(F['vh_early'], -99) >= LP_DRY_EARLY) & (_nz(F['vh_emin'], -99) > LP_DRY_EARLY - 2)
    else:                                  # F-4 fix 6: one (or <= 20 %) wet early scene does not make the spring wet
        dry = (_nz(F['vh_early'], -99) >= LP_DRY_EARLY) & (F['vh_ne21'] <= np.maximum(1, np.floor(EARLY_WET_SHARE * F['vh_ne'])))
    lp_rad = np.where((F['vh_n'] >= 4) & dry, _nz(ramp(F['vh_jmin2'], *LP_VH_RAMP), 0), 0.0)
    S['late_puddle'] = np.where(full >= 3, np.fmax(lp_opt, lp_rad), 0.0) if LATE_PUDDLE else np.zeros(len(full))
    lp_water = np.where(june, _nz(ramp(F['smin'], *LP_WATER_RAMP), 0), 0.0)     # clear open water in June, any size
    wet = np.fmax(np.fmax(S['flood_s1'], S['flood_R']), np.fmax(S['flood_opt'], S['late_puddle']))
    lpw = S['late_puddle']
    if 'barwet' not in OFF:
        # after a green spring (barley shape) low optical NDVI in May-June is usually bare soil after the barley harvest
        # (麦後大豆, NDVI 0.03-0.10): there only radar flooding, trend R or clear open water (NDVI <= 0) count as water
        # for the late-canopy gate, the wide harvest window and the late-green cancel
        bs = _nz(ramp(_first(F['bmax2'], F['A']), *BARLEY_RAMP), 0)
        lp_rad3 = np.where(full >= 3, lp_rad, 0.0)
        opt = np.fmax(S['flood_opt'], np.where(full >= 3, lp_opt, 0.0)) * (1 - bs)
        wet = np.fmax(np.fmax(np.fmax(S['flood_s1'], S['flood_R']), np.fmax(lp_rad3, lp_water)), opt)
        lpw = np.fmax(np.fmax(lp_rad3, lp_water), np.where(full >= 3, lp_opt, 0.0) * (1 - bs))
    S['water'] = wet
    # harvest: rice-time drop after the peak (F-4 fixes 4, 9: regrowth allowed; wider window with water evidence)
    hd = F['harv_doy'] if 'harv2' in OFF else F['harv2_doy']
    wetm = (wet >= 0.5) & ('harvwin' not in OFF)
    lo = np.where(wetm, HARV_WIN_WET[0], HARV_WIN[0]); hi = np.where(wetm, HARV_WIN_WET[1], HARV_WIN[1])
    pk, hm = F['peak'], F['harv_min']
    with np.errstate(invalid='ignore'):
        S['harvest'] = ((hd >= lo) & (hd <= hi) & (pk >= 0.6) & ((pk - hm) >= 0.25)).astype(float)
    # vigour: July canopy; a field with water evidence that greened up late may show its canopy in August - September
    vg = _first(F['ju_o'], F['jmax2'] - 0.05, F['J'])
    vig = _nz(ramp(vg, 0.45, 0.75), 0.5)
    if 'gate' in OFF:                      # F-2: late path x flood (then x flood again in flood_x_vig)
        gate = np.fmax(np.fmax(S['flood_s1'], S['flood_R']), S['flood_opt'])
    else:                                  # F-4 fix 2: saturating water gate
        gate = ramp(wet, *WATER_GATE)
    if 'latevig' in OFF:
        late_ok = _nz(F['green_doy'], 999) <= LATE_GREEN_DOY[0]; lv = _nz(F['aug_o'], 0)
    else:                                  # F-4 fix 5: canopy up to 9/20 and green-up by 8/18 when water evidence exists
        late_ok = _nz(F['green_doy'], 999) <= np.where(wet >= 0.5, LATE_GREEN_DOY[1], LATE_GREEN_DOY[0])
        lv = np.where(wet >= 0.5, np.fmax(_nz(F['aug_o'], 0), _nz(F['lenv'], 0)), _nz(F['aug_o'], 0))
    late = ramp(lv, 0.65, 0.8)
    if 'vigharv' not in OFF:               # water + rice-time harvest: a weaker canopy counts (0.50 -> 0.70)
        late = np.fmax(late, S['harvest'] * _nz(ramp(F['env2'], *VIG_HARV_RAMP), 0))
    S['vigour'] = np.fmax(vig, np.where(late_ok, gate * late, 0))
    S['bare_x_vig'] = S['bare'] * S['vigour']
    S['flood_x_vig'] = np.fmax(S['flood_s1'], S['flood_R']) * S['vigour']
    lg = _nz(ramp(F['sep2_o'], 0.55, 0.8), 0.4)
    S['late_green'] = lg if 'lglp' in OFF else lg * (1 - lpw)
    S['early_green'] = _nz(ramp(F['lmay_o'], 0.45, 0.6), 0) * (1 - S['late_puddle'])
    em = _first(F['emay_o'], bo)
    fl = np.fmax(S['flood_s1'], _nz(ramp(F['R'], -19.0, -22.0), 0))
    if 'bar2' in OFF:
        shape = _nz(ramp(np.fmax(F['A'], F['bmax']), 0.5, 0.7), 0)
    else:                                  # F-4 fix 10: 2nd highest NDVI 4/01-5/20 (spring weeds ~0.55 are not barley)
        shape = _nz(ramp(_first(F['bmax2'], F['A']), *BARLEY_RAMP), 0)
    S['barley'] = shape * _nz(ramp(F['bmin'], 0.4, 0.25), 0.5) * _nz(ramp(em, 0.33, 0.5), 0.5) * (1 - fl)
    if 'jw' not in OFF:                    # F-4 fix 7: barley then June flooding / open water + canopy = 麦後稲
        jrad = np.where(F['vh_n'] >= 4, _nz(ramp(F['vh_jmin2'], *LP_VH_RAMP), 0), 0.0)
        jw = np.fmax(np.where(full >= 3, jrad, 0.0), lp_water)
        S['barley'] = S['barley'] * (1 - jw * S['vigour'])
    # flooded weeds: radar flooding cancels "never bare" (not in tiny fields, whose radar is mixed with surroundings)
    gfl = np.where(full >= 3, np.fmax(np.fmax(S['flood_s1'], _nz(ramp(F['R'], -20.0, -23.0), 0)), S['late_puddle']), 0.0)
    S['grass'] = _nz(ramp(F['smin'], 0.28, 0.45), 0.5) * (1 - gfl)
    small = (full < 4).astype(float)
    S['small'] = small
    S['floodR_small'] = S['flood_R'] * small
    S['flood_s1_small'] = S['flood_s1'] * small
    # F-4 fix 3: a radar series without any water (radar, trend R, open water, late puddling) counts against rice
    wall = np.fmax(np.fmax(S['flood_s1'], _nz(ramp(F['R'], -19.0, -22.0), 0)),
                   np.fmax(np.fmax(S['flood_opt'], S['late_puddle']), lp_water))
    # only with a dense series (>= 12 scenes 4/20-7/10, 2025-: a short flood cannot fall between scenes) and >= 3 inner
    # pixels (the radar of smaller fields is mixed with the surroundings)
    S['no_water'] = np.where((F['vh_n'] >= NOWATER_MIN_N) & (full >= 3) & ('nowater' not in OFF), 1 - wall, 0.0)
    if 'nwbare' not in OFF:                # ... and never bare in May (issue 3: water = 0 and spring never bare)
        S['no_water'] = S['no_water'] * (1 - S['bare'])
    # F-4 fix 11: still green in October without a rice harvest (autumn optical only)
    S['noharv_green'] = (1 - S['harvest']) * _nz(ramp(F['oct_o'], 0.5, 0.7), 0) * ('nhg' not in OFF)
    return S


def barley_inputs(F):
    return dict(bA=np.clip(_nz(F['A'], 0.3), 0.1, 0.85),
                bMax=np.clip(_nz(np.fmax(F['bmax'], F['A']), 0.4), 0.2, 0.9),
                bMin=np.clip(_nz(F['bmin'], 0.4), 0.05, 0.6),
                bMay=np.clip(_nz(_first(np.fmin(F['emay_o'], F['may_o']), F['A']), 0.4), 0.1, 0.8),
                bJl=np.clip(_nz(F['jl_o'], 0.45), 0.1, 0.8))


def regime(F):
    """V1 = radar series (>= 4 scenes 4/20-7/10); P1 = autumn optical (>= 2 clear obs 10/1-11/30 or data to >= 10/20)."""
    V = (F['vh_n'] >= 4).astype(int)
    P = ((F['n_post'] >= 2) | (_nz(F['last_md'], 0) >= 1020)).astype(int)
    return np.array([f'V{a}P{b}' for a, b in zip(V, P)], dtype=object)


def _sig(z):
    return 1 / (1 + np.exp(-z))


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


# ================================================================== stage 1
KEEP = ['cell', 'pid', 'year', 'lon', 'lat', 'full', 'area_m2', 'R', 'vh_n', 'vh_frac', 'vh_q20', 'wmin', 'bare_o',
        'ju_o', 'aug_o', 'harv_doy', 'last_md', 'cov', 'hz']
COV_ALL = (1 << len(COVER)) - 1


def no_data(F):
    """Field-years without a single own-year observation (no clear optical obs 3/1-11/30, no radar scene, no trend
    A/J/R): the evidence would be all defaults, so they get no label instead of a confident 休 / 他."""
    return ((F['n_opt'] + F['n_pre'] + F['n_post'] == 0) & (F['vh_n_all'] == 0) &
            np.isnan(F['A']) & np.isnan(F['J']) & np.isnan(F['R']))


def stage1(F, model):
    """Own-year probabilities from a column dict.  Returns dict of arrays (p1, q, piR, piB, reg, S..., contrib)."""
    n = len(F['year'])
    S = evidence(F); Xb = barley_inputs(F); reg = regime(F)
    p1 = np.full(n, np.nan); q = np.full(n, np.nan); piR = np.full(n, np.nan); piB = np.full(n, np.nan)
    contrib = np.zeros((n, len(EVIDENCE)))
    for rname, m in model['stage1'].items():
        s = reg == rname
        if not s.any():
            continue
        z = np.full(int(s.sum()), m['b'])
        for c, wc in m['w'].items():
            x = S[c][s]; z = z + wc * x
            contrib[s, EVIDENCE.index(c)] = wc * x
        p1[s] = _sig(z)
        mb = model['barley'][rname]
        zb = np.full(int(s.sum()), mb['b'])
        for c, wc in mb['w'].items():
            zb = zb + wc * Xb[c][s]
        q[s] = _sig(zb)
        piR[s] = m['prior']; piB[s] = mb['prior']
    nd = no_data(F)
    p1[nd] = np.nan; q[nd] = np.nan
    out = {k: F[k] for k in KEEP}
    out.update(p1=p1, q=q, piR=piR, piB=piB, reg=reg, contrib=contrib,
               grass=S['grass'], harvest=S['harvest'], vigour=S['vigour'],
               flood=np.fmax(S['flood_s1'], S['flood_R']), water=S['water'])
    return out


def stage1_cell(cell, J, model):
    """Stage 1 for one cell: dict of arrays (one entry per field-year), or None when the cell has no fields."""
    rows = cell_features(cell, J)
    if not rows:
        return None
    return stage1(columns(rows), model)


def concat(parts):
    parts = [p for p in parts if p is not None]
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


# ================================================================== stage 2 (context)
def neighbour_pairs(lon, lat, radius):
    """All pairs (i < j) of centroids within `radius` m (grid hashing, numpy only).  NaN centroids have no pairs."""
    lon, lat = np.asarray(lon, float), np.asarray(lat, float)
    ok = np.isfinite(lon) & np.isfinite(lat)
    if not ok.all():
        iv = np.flatnonzero(ok)
        return iv[neighbour_pairs(lon[ok], lat[ok], radius)] if len(iv) else np.zeros((0, 2), np.int64)
    x = (lon - 136.0) * 111320 * np.cos(np.radians(LAT0)); y = (lat - LAT0) * 110950
    gx = np.floor(x / radius).astype(np.int64); gy = np.floor(y / radius).astype(np.int64)
    key = gx * 1000003 + gy
    order = np.argsort(key, kind='stable'); ks = key[order]
    out = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            nk = (gx + dx) * 1000003 + (gy + dy)
            lo = np.searchsorted(ks, nk, 'left'); hi = np.searchsorted(ks, nk, 'right')
            cnt = hi - lo
            if not cnt.any():
                continue
            i = np.repeat(np.arange(len(x)), cnt)
            j = order[np.concatenate([np.arange(a, b) for a, b in zip(lo[cnt > 0], hi[cnt > 0])])]
            ok = (i < j) & ((x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2 <= radius * radius)
            out.append(np.c_[i[ok], j[ok]])
    return np.concatenate(out) if out else np.zeros((0, 2), np.int64)


def _fb(E, T, init):
    NY = E.shape[1]
    al = np.zeros_like(E); be = np.ones_like(E); ll = np.zeros(E.shape[0])
    a = init * E[:, 0]; c = a.sum(1); al[:, 0] = a / c[:, None]; ll += np.log(c)
    for t in range(1, NY):
        a = (al[:, t - 1] @ T) * E[:, t]; c = a.sum(1); al[:, t] = a / c[:, None]; ll += np.log(c)
    for t in range(NY - 2, -1, -1):
        b = (E[:, t + 1] * be[:, t + 1]) @ T.T; be[:, t] = b / b.sum(1, keepdims=True)
    G = al * be
    return G / G.sum(-1, keepdims=True), ll


def hmm_mixture(E, Ts, inits, W, sc):
    Gs, lls = zip(*[_fb(E, np.asarray(T), np.asarray(i0)) for T, i0 in zip(Ts, inits)])
    L = np.stack(lls, 1) + np.log(np.asarray(W)[sc])
    L -= L.max(1, keepdims=True); R = np.exp(L); R /= R.sum(1, keepdims=True)
    return sum(R[:, k, None, None] * Gs[k] for k in range(len(Ts))), R


def stage2(R1, model, season_year=None, only_cells=None, context=True, exclude_incomplete=False):
    """Context + labels.  R1: stage-1 dict (concat of cells; include neighbour cells for the 150 m prior).
    season_year: the running season (its rice without a harvest is provisional), or None.
    Returns a dict of arrays, one entry per field-year of `only_cells` (default: all)."""
    C = model['context']
    pid = R1['pid']
    upid, fi = np.unique(pid, return_inverse=True)
    years = sorted(set(int(y) for y in R1['year']))
    ti = np.searchsorted(years, R1['year'])
    NF, NY = len(upid), len(years)
    first = np.full(NF, -1); first[fi[::-1]] = np.arange(len(fi))[::-1]

    def A(v, fill=np.nan):
        M = np.full((NF, NY), fill, dtype=float); M[fi, ti] = v
        return M
    P1 = A(R1['p1']); Q = A(R1['q']); PIR = A(R1['piR'], 0.67); PIB = A(R1['piB'], 0.35)
    miss = np.isnan(P1)
    # incomplete inputs (a per-year file not yet (re)fetched, a partial year, an unreadable file): flagged in the output
    # (reason suffix, `incomplete`; run_all --prev keeps the previous row).  exclude_incomplete=True also keeps such
    # years out of the rotation / neighbour context of the other field-years (off by default: in the simulations the
    # lost context moved about as many labels of complete years as the degraded evidence did)
    inc = A(R1['cov'], COV_ALL) != COV_ALL if 'cov' in R1 else np.zeros_like(miss)
    inc &= ~miss
    hz = A(R1['hz'], 1231) if 'hz' in R1 else np.full(miss.shape, 1231.0)
    # a running year whose data end before 9/01 is mostly defaults: it must not move the finished years' labels
    exc = ((inc if exclude_incomplete else np.zeros_like(inc)) | (hz < SEASON_LATE_MD)) & ~miss
    S1 = np.stack([P1, (1 - P1) * Q, (1 - P1) * (1 - Q)], -1)
    PI = np.stack([PIR, (1 - PIR) * PIB, (1 - PIR) * (1 - PIB)], -1)
    PF = P1.copy(); G = S1.copy(); d = np.zeros_like(P1)
    if context:
        E = S1 / PI; E[miss | exc] = 1.0; E = E / E.sum(-1, keepdims=True)
        full = _nz(R1['full'][first], 1)
        sc = np.digitize(full, SZ_BINS)
        pairs = neighbour_pairs(R1['lon'][first], R1['lat'][first], C['radius_m'])
        S1n = np.nan_to_num(S1)
        st = np.argmax(S1n, -1); conf = (S1n.max(-1) >= C['conf']) & ~miss & ~exc
        cnt = np.zeros((NF, NY, 3))
        if len(pairs):
            i, j = pairs[:, 0], pairs[:, 1]
            for t in range(NY):
                for a, b in ((i, j), (j, i)):
                    c = conf[b, t]
                    np.add.at(cnt[:, t, :], (a[c], st[b[c], t]), 1.0)
        glob = np.asarray(C['glob'])
        ls = np.log((cnt + C['a0'] * glob) / (cnt.sum(-1, keepdims=True) + C['a0']))
        oh = np.eye(len(SZ_BINS) + 1)[np.repeat(sc[:, None], NY, 1)]
        X = np.concatenate([ls, oh, ls[..., :1] * oh], -1).reshape(NF * NY, -1)
        nm = C['nbr']
        Z = X @ np.asarray(nm['coef']).T + np.asarray(nm['intercept'])
        Z -= Z.max(1, keepdims=True); Pn = np.exp(Z); Pn /= Pn.sum(1, keepdims=True)
        Pn = Pn.reshape(NF, NY, 3)
        Nr = Pn / np.asarray(nm['prior'])[np.repeat(sc[:, None], NY, 1)]
        EN = E * Nr ** C['kappa']; EN[exc] = 1.0
        G, _ = hmm_mixture(EN, C['Ts'], C['inits'], C['W'], sc)
        l1 = _logit(P1)
        d = np.clip(_logit(G[..., 0]) - l1, -C['clip'], C['clip'])
        # an incomplete year was left out of E, so the HMM posterior is prior x context only; add its own evidence back
        d = np.where(exc, np.clip(_logit(G[..., 0]) - _logit(PIR), -C['clip'], C['clip']), d)
        PF = _sig(l1 + d)
    PF[miss] = np.nan
    qb = G[..., 1] / np.clip(G[..., 1] + G[..., 2], 1e-9, None)
    grass = A(R1['grass'], 0); harv = A(R1['harvest'], 0); vig = A(R1['vigour'], 0.5); fl = A(R1['flood'], 0)
    rice = PF >= 0.5
    barley = ~rice & (qb >= 0.5) & (Q >= 0.25)
    # 休 = never tilled (green all spring), flooded but no crop (調整水田), or green in July without a rice harvest
    # (weeds / mowed grass / 保全管理).  他 = the rest: soybean / soba sown in summer (low in July), vegetables, bare.
    fallow = ~rice & ~barley & (((grass >= 0.5) & (harv < 0.5)) | ((fl >= 0.6) & (vig < 0.3)) |
                                ((vig >= FALLOW_VIG) & (harv < 0.5)))
    lab = np.where(rice, '稲', np.where(barley, '麦', np.where(fallow, '休', '他')))
    lab[miss] = ''                                   # no own-year data (or year absent for this field): no label
    # provisional: the running season (calendar) and any year whose data does not yet reach 10/20 (e.g. a run in
    # November while the data pipeline is behind); all its labels while the data end before 9/01 (early season)
    run = hz < SEASON_CLOSED_MD
    if season_year in years:
        run[:, years.index(season_year)] = True
    prov = run & ~miss & ((hz < SEASON_LATE_MD) | (rice & (harv < 0.5)) | ((PF > 0.2) & (PF < 0.8)))
    # back to the record order of R1
    sel = np.ones(len(fi), bool) if only_cells is None else np.isin(R1['cell'], list(only_cells))
    f_, t_ = fi[sel], ti[sel]
    out = dict(cell=R1['cell'][sel], pid=R1['pid'][sel], year=R1['year'][sel], label=lab[f_, t_],
               p_rice=PF[f_, t_], provisional=prov[f_, t_], p1=P1[f_, t_], d=d[f_, t_],
               flood=fl[f_, t_], vigour=vig[f_, t_], grass=grass[f_, t_], incomplete=inc[f_, t_])
    out['reason'] = reasons(out, R1['contrib'][sel])
    for k in ('reg', 'R', 'vh_frac', 'wmin', 'bare_o', 'ju_o', 'aug_o', 'harv_doy', 'area_m2', 'full'):
        out[k] = R1[k][sel]
    return out


def reasons(o, contrib):
    """Short Japanese reason: the two largest weighted evidence terms + context note + provisional note."""
    ja = [EVIDENCE_JA[n] for n in EVIDENCE]
    ib = EVIDENCE.index('barley')
    res = np.empty(len(o['label']), dtype=object)
    for k in range(len(res)):
        L = o['label'][k]; c = contrib[k]; p1 = o['p1'][k]; pf = o['p_rice'][k]
        if np.isnan(p1):
            res[k] = 'データなし'; continue
        if L == '稲':
            ev = [ja[i] for i in np.argsort(-c)[:2] if c[i] > 0.5]
            txt = '稲: ' + ('・'.join(ev) if ev else '弱い証拠')
        elif L == '麦':
            txt = '麦: 4月に緑→6月前後に刈取り'
        elif L == '休':
            if o['flood'][k] >= 0.6 and o['vigour'][k] < 0.3:
                txt = '休: 湛水のみで生育なし'
            elif o['grass'][k] >= 0.5:
                txt = '休: 春から緑・耕起なし'
            else:
                txt = '休: 夏も草で緑・刈取りなし'
        else:
            ev = [ja[i] for i in np.argsort(c)[:2] if c[i] < (-4.0 if i == ib else -1.0)]
            ms = []
            if o['flood'][k] < 0.2:
                ms.append('湛水なし')
            if o['vigour'][k] < 0.4:
                ms.append('夏の生育弱い')
            txt = '他: ' + '・'.join((ev + ms)[:2] or ['稲の証拠が弱い'])
        if (pf >= 0.5) != (p1 >= 0.5):
            txt += '／前後年・周辺から' + ('稲と判定' if pf >= 0.5 else '稲でないと判定')
        elif 0.1 < p1 < 0.9 and abs(o['d'][k]) >= 0.4:
            txt += '／前後年・周辺も' + ('稲寄り' if o['d'][k] > 0 else '非稲寄り')
        if o['provisional'][k]:
            txt += '（暫定: 収穫前）' if L == '稲' else '（暫定）'
        if o['incomplete'][k]:
            txt += '（データ不足）'
        res[k] = txt
    return res


def label_cell(cell, J, model, neighbour_records=None, season_year=None):
    """Label one cell.  neighbour_records: stage-1 dicts of adjacent cells (optional; improves the 150 m prior at
    the cell edge).  Returns the stage-2 dict for this cell's field-years."""
    own = stage1_cell(cell, J, model)
    if own is None:
        return None
    R1 = concat([own] + list(neighbour_records or []))
    return stage2(R1, model, season_year, only_cells=[cell])


# ================================================================== output
def _r(x, nd=2):
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else round(float(x), nd)


def _mmdd(doy, year):
    if doy is None or np.isnan(doy):
        return None
    d = np.datetime64(f'{int(year)}-01-01') + int(doy) - 1
    return str(d)[5:]


OUT_KEYS = ['label', 'p_rice', 'provisional', 'reason', 'p_own', 'flood', 'R', 'vh_frac', 'water', 'bare', 'jul',
            'aug', 'harvest', 'regime']


def cell_index(o):
    """{cell: record indices} in one pass (to_crop_json with `idx` avoids a scan of all records per cell)."""
    order = np.argsort(o['cell'], kind='stable'); cs = o['cell'][order]
    cut = np.flatnonzero(cs[1:] != cs[:-1]) + 1
    return {str(g[0]): order[a:b] for g, a, b in zip(np.split(cs, cut), np.r_[0, cut], np.r_[cut, len(cs)])} if len(cs) else {}


def to_crop_json(o, cell, idx=None, override=None):
    """crop.json for one cell: {"v", "keys", "p": {pid: {year: [values in `keys` order]}}}.
    idx: record indices of the cell (cell_index); override: {(pid, year): row} rows kept from the previous crop.json."""
    ks = np.flatnonzero(o['cell'] == cell) if idx is None else idx
    P = {}
    for k in ks:
        y = int(o['year'][k])
        if override and (str(o['pid'][k]), y) in override:
            P.setdefault(str(o['pid'][k]), {})[str(y)] = override[(str(o['pid'][k]), y)]; continue
        P.setdefault(str(o['pid'][k]), {})[str(y)] = [
            str(o['label'][k]), _r(o['p_rice'][k]), bool(o['provisional'][k]), str(o['reason'][k]),
            _r(o['p1'][k]), _r(o['flood'][k]), _r(o['R'][k], 1), _r(o['vh_frac'][k]), _r(o['wmin'][k]),
            _r(o['bare_o'][k]), _r(o['ju_o'][k]), _r(o['aug_o'][k]), _mmdd(o['harv_doy'][k], y), str(o['reg'][k])]
    return dict(v=VERSION, keys=OUT_KEYS, p=P)


def to_crop_slim(full_json):
    """Viewer file (crop_s.json): label, p_rice (%), provisional, reason as an index into a per-cell reason table.
    {"v", "years": [..], "keys": ["label", "p", "prov", "r"], "reasons": [...], "p": {pid: [[L, p, prov, r] | null per year]}}"""
    P = full_json['p']
    years = sorted({int(y) for d in P.values() for y in d})
    rt, ri, out = [], {}, {}
    for pid, d in P.items():
        row = []
        for y in years:
            v = d.get(str(y))
            if v is None:
                row.append(None); continue
            r = v[3]
            if r not in ri:
                ri[r] = len(rt); rt.append(r)
            row.append([v[0], None if v[1] is None else int(round(v[1] * 100)), int(bool(v[2])), ri[r]])
        out[pid] = row
    return dict(v=full_json['v'], years=years, keys=['label', 'p', 'prov', 'r'], reasons=rt, p=out)
