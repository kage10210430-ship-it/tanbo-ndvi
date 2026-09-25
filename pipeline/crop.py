"""Label every field-year of all cells (稲・麦・休・他): stage 1 per cell (4 processes), stage 2 across all cells, then write
<out>/cells/<cell>/crop.json and the slim viewer file crop_s.json (Actions: --data site/data --out site/data --prev site/data).

  python pipeline/crop.py [--data DIR] [--out DIR] [--cells a,b] [--season-year 2026 | --today YYYY-MM-DD] [--dump labels.pkl]
                     [--prev DIR] [--no-slim] [--no-full]

--prev DIR: the previously published output (DIR/cells/<cell>/crop.json).  Finished field-years whose inputs are
incomplete now (a per-year file missing / partial / unreadable, e.g. while trend.py back-fills a year that left the
rolling ndvi.json) keep their previous row when it was made by the same classifier version.
Exit status 2 when more than 5 % of the cells failed (their crop.json files are left as they were).
"""
import argparse, datetime, json, os, pickle, sys, time
from multiprocessing import Pool
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import crop_core as CC

DATA = os.path.join(os.path.dirname(HERE), 'site', 'data')
JST = datetime.timezone(datetime.timedelta(hours=9))
_MODEL = None


def _job(args):
    cell, data = args
    global _MODEL
    if _MODEL is None:
        _MODEL = CC.load_model()
    try:
        J = CC.load_cell(os.path.join(data, 'cells', cell))
        return cell, CC.stage1_cell(cell, J, _MODEL), J['_errors']
    except Exception as e:                       # one broken cell must not stop the other 636
        return cell, None, [f'stage1 {type(e).__name__}: {e}']


def season_of(today):
    """The running season: this calendar year until the end of October (late rice is harvested by mid-October).
    Years whose data does not reach 10/20 yet stay provisional after that (crop_core.stage2, `closed`)."""
    return today.year if today.month <= 10 else None


def _write(path, obj):
    """Atomic write: a killed job never leaves a half-written crop.json for the viewer."""
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as fh:
        json.dump(obj, fh, ensure_ascii=False, separators=(',', ':'))
    os.replace(tmp, path)


def _prev_rows(prev_dir, cell, want):
    """{(pid, year): row} from the previous crop.json of `cell` for the (pid, year) pairs in `want` (same version).
    Without crop.json (published data carry only crop_s.json) the slim rows are expanded: label, p, provisional, reason."""
    d = os.path.join(prev_dir, 'cells', cell)
    try:
        with open(os.path.join(d, 'crop.json'), encoding='utf-8') as fh:
            pj = json.load(fh)
    except (OSError, ValueError):
        pj = None
    if pj is not None:
        if pj.get('v') != CC.VERSION or pj.get('keys') != CC.OUT_KEYS:
            return {}
        P = pj.get('p', {})
        return {(p, y): P[p][str(y)] for p, y in want if str(y) in P.get(p, {})}
    try:
        with open(os.path.join(d, 'crop_s.json'), encoding='utf-8') as fh:
            sj = json.load(fh)
    except (OSError, ValueError):
        return {}
    if sj.get('v') != CC.VERSION:
        return {}
    ys, P, rt, out = [str(y) for y in sj.get('years', [])], sj.get('p', {}), sj.get('reasons', []), {}
    for p, y in want:
        if p in P and str(y) in ys:
            v = P[p][ys.index(str(y))]
            if v:
                out[(p, y)] = [v[0], None if v[1] is None else v[1] / 100, bool(v[2]), rt[v[3]] if v[3] < len(rt) else ''] + \
                              [None] * (len(CC.OUT_KEYS) - 5) + ['']
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=DATA)
    ap.add_argument('--out', default=DATA)
    ap.add_argument('--cells', default='')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--season-year', type=int, default=None)
    ap.add_argument('--today', default='')
    ap.add_argument('--dump', default='', help='also pickle the stage-2 arrays (for the harness)')
    ap.add_argument('--prev', default='', help='previous output dir: keep rows of finished years with incomplete inputs')
    ap.add_argument('--no-slim', action='store_true', help='do not write crop_s.json')
    ap.add_argument('--no-full', action='store_true', help='do not write crop.json (the published site keeps only crop_s.json)')
    ap.add_argument('--no-context', action='store_true')
    a = ap.parse_args()
    t0 = time.time()
    cells = a.cells.split(',') if a.cells else [c['id'] for c in json.load(open(os.path.join(a.data, 'index.json')))['cells']]
    # the date in Japan (Actions runs at 19:23 UTC = 4:23 JST of the next day)
    today = datetime.date.fromisoformat(a.today) if a.today else datetime.datetime.now(JST).date()
    season = a.season_year if a.season_year else season_of(today)
    with Pool(a.workers) as p:
        res = p.map(_job, [(c, a.data) for c in cells], chunksize=2)
    failed = [c for c, r, e in res if r is None and any(x.startswith('stage1') for x in e)]
    for c, r, e in res:
        for x in e:
            print(f'::warning::{c}: {x}')
    parts = [r for c, r, e in res if r is not None]
    t1 = time.time()
    if not parts:
        print('::error::no field-years in any cell'); sys.exit(2)
    model = CC.load_model()
    R1 = CC.concat(parts)
    o = CC.stage2(R1, model, season, context=not a.no_context)
    t2 = time.time()
    idx = CC.cell_index(o)
    done = set(idx)
    kept = 0
    for c in cells:
        if c not in done:
            if c in failed:
                continue                          # keep the old file of a failed cell
            ks = np.zeros(0, int)
        else:
            ks = idx[c]
        over = None
        if a.prev and len(ks):
            fin = ks[o['incomplete'][ks] & ~o['provisional'][ks]]
            if len(fin):
                over = _prev_rows(a.prev, c, [(str(o['pid'][k]), int(o['year'][k])) for k in fin])
                for k in fin:
                    row = (over or {}).get((str(o['pid'][k]), int(o['year'][k])))
                    if row is not None:
                        o['label'][k], o['p_rice'][k], o['reason'][k] = row[0], np.nan if row[1] is None else row[1], row[3]
                        kept += 1
        d = os.path.join(a.out, 'cells', c); os.makedirs(d, exist_ok=True)
        cj = CC.to_crop_json(o, c, ks, over)
        if not a.no_full:
            _write(os.path.join(d, 'crop.json'), cj)
        if not a.no_slim:
            _write(os.path.join(d, 'crop_s.json'), CC.to_crop_slim(cj))
    t3 = time.time()
    if a.dump:
        with open(a.dump, 'wb') as fh:
            pickle.dump({k: o[k] for k in ('pid', 'year', 'label', 'p_rice', 'provisional', 'reason', 'p1', 'reg', 'cell',
                                            'incomplete')}, fh)
    ys, labs = o['year'], o['label']
    print(f'{len(labs)} field-years, {len(cells)} cells ({len(failed)} failed), season {season}: stage1 {t1 - t0:.0f}s  '
          f'stage2 {t2 - t1:.0f}s  write {t3 - t2:.0f}s  total {time.time() - t0:.0f}s')
    for y in sorted(set(ys.tolist())):
        s = ys == y
        regs = dict(zip(*np.unique(o['reg'][s].astype(str), return_counts=True)))
        print(y, {L: int((labs[s] == L).sum()) for L in ('稲', '麦', '休', '他', '')}, 'provisional', int(o['provisional'][s].sum()),
              'incomplete', int(o['incomplete'][s].sum()), 'regimes', {str(k): int(v) for k, v in regs.items()})
        if o['incomplete'][s].any():
            ic = sorted(set(o['cell'][s & o['incomplete']].tolist()))
            print(f'::warning::{y}: {int(o["incomplete"][s].sum())} field-years in {len(ic)} cells have incomplete inputs '
                  f'(e.g. {", ".join(ic[:5])})' + ('' if a.prev else '; pass --prev to keep the old rows'))
    if a.prev:
        print(f'{kept} field-years with incomplete inputs kept their previous row (--prev {a.prev})')
    if len(failed) > 0.05 * len(cells):
        print(f'::error::{len(failed)} of {len(cells)} cells failed'); sys.exit(2)


if __name__ == '__main__':
    main()
