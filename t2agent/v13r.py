"""v13r (claude2-lab T2 release lane, 2026-10-13; v13r3 2026-10-14): official-data release cards (weekly / monthly / quarterly series, horizon <= 20 BD).

The baseline's step count and 300-step drift are wrong for these cards (a quarterly series is walked 3 "monthly" steps, a weekly
claims series is walked on a window that still contains the 2021 normalisation, an index is walked with a 25-year mean drift).
v13r replaces centre, width and shape; the baseline is still used for nothing but the fallback path in core.forecast.

  steps   true number of series periods from the last panel observation to the target period (named month > card target date >
          as-of + h business days; target period = period before the one that contains the release date; a weekly series whose last
          row is at most 20 days old moves one week per 7 calendar days from the as-of)
  centre  inverse-MAE^4 combination of window-mean predictors on log levels (positive series that are large or move < 3% per step)
          or levels: last + k-step mean drift
          (k = 2,4,8,12,24) and mean of the last w values (w = 1,2,4,8,13,26); weights from the last N one-step self-errors;
          plus x[t-P] + last P-period change (P = 52 / 12 / 4) when the lag-P autocorrelation of the last 11 P changes exceeds 0.3
          (an unadjusted series; never fires on the adjusted test series)
  width   1.4826 MAD of the combined self-errors x 1.15 (floor 0.3 RMS), Student-t(4) shape on an exact quantile grid
  PCE     PCE cards whose panel also holds the matching CPI series for the missing month(s) (PCE->CPI in the id, FRED codes, else the
          CPI id of the same core-ness with most tokens in common; same scale): Huber regression of the PCE (log) change on the CPI
          (log) change over the last 60 consecutive months, or for a level-scale series (rates) also of level on level, whichever
          implies the smaller sd (width 1.3 x residual MAD-sd); the engine otherwise
  joint   Gaussian copula: same asset = random-walk correlation sqrt(s_i/s_j) (identical columns when the steps agree),
          different assets = shrunk correlation of the self-errors; draws placed by the ranks of correlated normals
"""
import re
import zlib
import numpy as np
import pandas as pd

N_DRAWS = 500
M_ENG = {'W': 1.15, 'M': 1.15, 'Q': 1.15}
M_CPI = 1.3
NWIN = {'W': 104, 'M': 60, 'Q': 40}
DRIFT_K = (2, 4, 8, 12, 24)
MEAN_W = (1, 2, 4, 8, 13, 26)
MAX_H = 20
MAX_S = {'W': 13, 'M': 6, 'Q': 4}
PERIOD = {'W': 52, 'M': 12, 'Q': 4}
SEAS_RHO = 0.3
_GRID = {}


def t4_grid(n=N_DRAWS):
    """exact Student-t (4 dof) quantiles at (k+.5)/n, scale 1"""
    if n not in _GRID:
        p = (np.arange(n) + 0.5) / n
        a = 4 * p * (1 - p)
        q = np.cos(np.arccos(np.sqrt(a)) / 3) / np.sqrt(a)
        _GRID[n] = np.sign(p - 0.5) * 2 * np.sqrt(np.maximum(q - 1, 0))
    return _GRID[n]


def mad_sd(e):
    e = np.asarray(e, float)
    e = e[np.isfinite(e)]
    return 1.4826 * float(np.median(np.abs(e - np.median(e)))) if len(e) else 0.0


def cadence(index):
    """('W' | 'M' | 'Q' | None, median spacing in days) from the last 12 gaps of a DatetimeIndex"""
    if len(index) < 8:
        return None, np.nan
    d = float(np.median(np.diff(index[-13:]).astype('timedelta64[D]').astype(float)))
    if 5 <= d <= 9:
        return 'W', d
    if 26 <= d <= 33:
        return 'M', d
    if 84 <= d <= 95:
        return 'Q', d
    return None, d


def _pidx(ts, cls):
    return ts.year * 12 + ts.month if cls == 'M' else ts.year * 4 + (ts.month - 1) // 3


def steps_for(last, cls, h, asof, target_date=None, named=None):
    """periods between the last panel observation and the target period"""
    if named is not None and cls in 'MQ':
        s = _pidx(named, cls) - _pidx(last, cls)
        if s >= 1:
            return s
    T = target_date if target_date is not None else pd.Timestamp(asof) + pd.offsets.BDay(int(h))
    if cls == 'W':
        if (pd.Timestamp(asof) - last).days <= 20:
            return max(int(np.ceil((T - pd.Timestamp(asof)).days / 7.0)), 1)
        dd = (T - last).days
        s = dd // 7 if T.weekday() == last.weekday() else int(np.floor((dd - 1) / 7.0))
    else:
        s = _pidx(T, cls) - 1 - _pidx(last, cls)
    return max(int(s), 1)


def _preds(x, t, s, dk, mw, P=0):
    """candidate predictions of x[t] from x[:t-s+1] (s steps ahead); P > 0 adds the seasonal x[t-P] + last P-period change"""
    o = t - s
    out = [x[o] + s * (x[o] - x[o - k]) / k for k in dk]
    out += [x[o - w + 1:o + 1].mean() for w in mw]
    if P:
        out.append(x[t - P] + x[o] - x[o - P])
    return out


def _now(x, s, dk, mw, P=0):
    out = [x[-1] + s * (x[-1] - x[-1 - k]) / k for k in dk]
    out += [x[-w:].mean() for w in mw]
    if P:
        out.append(x[len(x) - 1 + s - P] + x[-1] - x[-1 - P])
    return out


def seasonal(x, P):
    """True if the lag-P autocorrelation of the last ~11 periods of changes exceeds SEAS_RHO (an unadjusted series)"""
    if not P or len(x) <= 3 * P + 2:
        return False
    dz = np.diff(np.asarray(x, float))[-11 * P:]
    a, b = dz[P:], dz[:-P]
    if np.std(a) == 0 or np.std(b) == 0:
        return False
    return bool(np.corrcoef(a, b)[0, 1] > SEAS_RHO)


def engine(x, s, N, P=0):
    """x: log level (positive series) or level, oldest first. Returns (centre, one-step-style self-errors, weights) for an s-step forecast.
    Candidate windows are dropped from the long end until at least 12 self-forecast origins exist."""
    x = np.asarray(x, float)
    n = len(x)
    dk, mw = list(DRIFT_K), list(MEAN_W)
    P = P if (P and s <= P and seasonal(x, P)) else 0
    while True:
        need = max([k for k in dk] + [w - 1 for w in mw] + [1, P])
        Nn = min(N, n - s - need)
        if Nn >= 12 or (len(dk) + len(mw)) <= 3:
            break
        if dk and (not mw or max(dk) >= max(mw)):
            dk.pop()
        else:
            mw.pop()
    if Nn < 12:
        raise ValueError('series too short')
    P_ = np.array([_preds(x, t, s, dk, mw, P) for t in range(n - Nn, n)])
    act = x[n - Nn:]
    mae = np.abs(P_ - act[:, None]).mean(0)
    wt = (mae.min() / np.maximum(mae, 1e-12)) ** 4
    wt = np.where(wt < 1e-6, 0.0, wt)
    wt /= wt.sum()
    err = act - (P_ * wt[None, :]).sum(1)
    c = float((np.array(_now(x, s, dk, mw, P)) * wt).sum())
    return c, err, wt


def _scale(err, m, floor):
    e = np.asarray(err, float)
    return max(mad_sd(e) * m, 0.3 * float(np.sqrt(np.mean(e ** 2))), floor)


def _huber(X, y, iters=10):
    w = np.ones(len(y))
    for _ in range(iters):
        b = np.linalg.lstsq(X * w[:, None], y * w, rcond=None)[0]
        r = y - X @ b
        s = max(mad_sd(r), 1e-12)
        w = np.sqrt(np.minimum(1.0, 1.5 * s / np.maximum(np.abs(r), 1e-12)))
    return b, r


_FRED_CPI = {'pcepi': ('cpiaucsl', 'cpiaucns'), 'pcepilfe': ('cpilfesl', 'cpilfens')}
_CORE = re.compile(r'(?i)core|lfe|ex[_ ]?f|xfe|less[_ ]?f')
PARTNER_MODE = 'auto'


def _toks(x):
    return {t for t in re.split(r'[^a-z0-9]+', re.sub(r'pce|cpi|core|lfe|ex_?food_?(and_?)?energy|xfe', ' ', str(x).casefold())) if t}


def _partner(asset, panels, asof):
    """CPI series matching a PCE target, as an as-of history or None: PCE->CPI substitution in the id, FRED codes, else the panel's
    CPI series of the same core-ness with the most id tokens in common"""
    a = str(asset)
    if not re.search(r'(?i)pce', a):
        return None
    from . import core
    cands = [re.sub(r'(?i)pce', lambda m: 'CPI' if m.group(0).isupper() else 'cpi', a)]
    cands += list(_FRED_CPI.get(a.strip().casefold(), ()))
    ids = sorted({str(x) for _, df in panels for x in df['asset'].unique()})
    core_t = bool(_CORE.search(a))
    tk = _toks(a)
    scan = [x for x in ids if 'cpi' in x.casefold() and bool(_CORE.search(x)) == core_t]
    cands += sorted(scan, key=lambda x: (-len(_toks(x) & tk), len(_toks(x) ^ tk), x))
    for cand in cands:
        try:
            h = core.history(panels, cand, asof)
            if len(h) >= 37:
                return h
        except Exception:
            continue
    return None


def cell_forecast(hist, s, cls, partner=None):
    """one cell: returns (fn(q) -> level draws for a standardised t-grid vector q, info, self-error Series aligned by date)"""
    x = np.asarray(hist.to_numpy(), float)
    x = x[-400:]
    idx = hist.index[-400:]
    use_log = bool(np.all(x > 0)) and (float(np.max(x)) >= 25 or float(np.median(np.abs(np.diff(x)))) < 0.03 * float(np.median(x)))
    z = np.log(x) if use_log else x
    N = NWIN[cls]
    c, err, wt = engine(z, s, N, PERIOD[cls])
    sc = _scale(err, M_ENG[cls], 1e-4 if use_log else 1e-4 * max(abs(c), 1.0))
    info = dict(s=s, cls=cls, log=use_log, model='engine')
    if partner is not None and cls == 'M':
        try:
            pc = partner[partner > 0] if use_log else partner
            if use_log and (len(pc) < len(partner) or float(pc.max()) < 25):
                raise ValueError('partner not on a log scale')
            if not use_log:
                ra = float(np.median(np.abs(pc.to_numpy()))) / max(float(np.median(np.abs(x))), 1e-12)
                if not 0.2 <= ra <= 5:
                    raise ValueError('partner scale')
            zp = pd.Series(z, index=idx.year * 12 + idx.month - 1)
            zc = pd.Series(np.log(pc.to_numpy()) if use_log else pc.to_numpy(), index=pc.index.year * 12 + pc.index.month - 1)
            zp = zp[~zp.index.duplicated(keep='last')]
            zc = zc[~zc.index.duplicated(keep='last')]
            last = zp.index[-1]
            miss = [last + j for j in range(1, s + 1)]
            if all(mo in zc.index for mo in miss):
                fits = []
                for lvl in ((False, True) if (not use_log and PARTNER_MODE != 'diff') else (False,)):
                    if lvl:
                        yv, xv = zp, zc
                    else:
                        yv = zp.diff()[np.diff(zp.index.to_numpy(), prepend=-9) == 1]
                        xv = zc.diff()[np.diff(zc.index.to_numpy(), prepend=-9) == 1]
                    fit = yv.index.intersection(xv.index).sort_values()[-60:]
                    if len(fit) < 36:
                        continue
                    X = np.column_stack([np.ones(len(fit)), xv.loc[fit].to_numpy()])
                    b, res = _huber(X, yv.loc[fit].to_numpy())
                    if lvl:
                        c_ = b[0] + b[1] * float(zc.loc[miss[-1]])
                        sd = max(mad_sd(res) * M_CPI, 0.3 * float(np.sqrt(np.mean(res ** 2))), 1e-4 * max(abs(z[-1]), 1.0))
                    else:
                        c_ = z[-1] + sum(b[0] + b[1] * float(xv.loc[mo]) if mo in xv.index else b[0] for mo in miss)
                        sd = max(mad_sd(res) * M_CPI * np.sqrt(s), 0.3 * float(np.sqrt(np.mean(res ** 2))) * np.sqrt(s),
                                 1e-4 if use_log else 1e-4 * max(abs(z[-1]), 1.0))
                    fits.append((float(sd), float(c_), lvl))
                if fits:
                    if PARTNER_MODE == 'lvl':
                        fits = fits[-1:]
                    sd, c_, lvl = min(fits)
                    c, sc = c_, sd
                    info['model'] = 'cpi-regression' + ('-lvl' if lvl else '')
        except Exception:
            pass
    f = (lambda q: np.exp(c + sc * q)) if use_log else (lambda q: c + sc * q)
    return f, info, pd.Series(err, index=idx[-len(err):])


def _ranks_from_corr(R, rng, n):
    d = R.shape[0]
    try:
        L = np.linalg.cholesky(R + 1e-9 * np.eye(d))
    except np.linalg.LinAlgError:
        L = np.diag(np.ones(d))
    Z = rng.standard_normal((n, d)) @ L.T
    return np.argsort(np.argsort(Z, axis=0), axis=0)


def applicable(p, card, spec, panels, asof):
    """True if every cell is a weekly / monthly / quarterly series with a short (<= 20 BD) horizon"""
    if p['ttype'] != 'level':
        return False
    if max(h for _, h in p['cells']) > MAX_H:
        return False
    for a in p['assets']:
        cls, _ = cadence(p['hist'][a].index)
        if cls is None or len(p['hist'][a]) < 16:
            return False
    return True


def draws(p, card, spec, panels, asof, unit_id):
    from . import core
    cells = p['cells']
    d = len(cells)
    asof = pd.Timestamp(asof)
    tds = core.target_dates(card, spec, asof, cells)
    fn, info, errs, steps = [], [], {}, []
    for (a, h) in cells:
        hist = p['hist'][a]
        cls, _ = cadence(hist.index)
        td, is_named = tds[(a, h)]
        named = td if is_named else None
        tdate = td if (td is not None and not is_named) else None
        s = steps_for(hist.index[-1], cls, h, asof, tdate, named)
        if s > MAX_S[cls]:
            raise ValueError(f'{a}: {s} steps')
        partner = _partner(a, panels, asof) if cls == 'M' else None
        f, inf, er = cell_forecast(hist, s, cls, partner)
        fn.append(f); info.append(inf); steps.append(s)
        errs.setdefault(a, er)
    R = np.eye(d)
    for i in range(d):
        for j in range(i + 1, d):
            (ai, hi), (aj, hj) = cells[i], cells[j]
            if ai == aj:
                r = np.sqrt(min(steps[i], steps[j]) / max(steps[i], steps[j]))
            else:
                ei, ej = errs[ai], errs[aj]
                com = ei.index.intersection(ej.index)
                r = 0.0
                if len(com) >= 20:
                    r = float(np.corrcoef(ei.loc[com].to_numpy(), ej.loc[com].to_numpy())[0, 1])
                    r = 0.7 * r if np.isfinite(r) else 0.0
            R[i, j] = R[j, i] = r
    rng = np.random.default_rng(zlib.crc32(unit_id.encode()) & 0x7FFFFFFF)
    q = t4_grid()
    if d == 1:
        X = fn[0](q)[:, None]
    else:
        rk = _ranks_from_corr(R, rng, N_DRAWS)
        X = np.column_stack([fn[j](q[rk[:, j]]) for j in range(d)])
    if not np.isfinite(X).all():
        raise ValueError('non-finite draws')
    return X, info, steps
