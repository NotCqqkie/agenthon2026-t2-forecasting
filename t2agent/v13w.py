"""v13w (claude2-lab T2 swarm, 2026-10-11): short-horizon UST / G10 FX draws for the ranked Final.

On top of v13v (v12 centre + M0 joint Gaussian deviations), four levers, each confirmed on blind holdout cards:
  centre  UST keeps the v12 centre; FX centre = anchor + 0.25 * (v12 centre - anchor)
  scale   per asset clip((EWMA(0.94) sd of daily changes / sd of the last 300 changes)^0.8, 0.5, 2) replaces (v60/v300)^0.75
  events  event-timed variance by horizon: FOMC (lull and jump), US payrolls, BoJ, ECB (scheduled calendars, known in advance)
  grid    each cell's 500 draws are the exact Gaussian quantile grid with the model sd, placed by the ranks of M0's draws
          (keeps M0's joint rank structure, removes 500-draw sampling noise)
"""
import zlib
import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
from statistics import NormalDist

_F = {
    2000: ['02-02', '03-21', '05-16', '06-28', '08-22', '10-03', '11-15', '12-19'],
    2001: ['01-31', '03-20', '05-15', '06-27', '08-21', '10-02', '11-06', '12-11'],
    2002: ['01-30', '03-19', '05-07', '06-26', '08-13', '09-24', '11-06', '12-10'],
    2003: ['01-29', '03-18', '05-06', '06-25', '08-12', '09-16', '10-28', '12-09'],
    2004: ['01-28', '03-16', '05-04', '06-30', '08-10', '09-21', '11-10', '12-14'],
    2005: ['02-02', '03-22', '05-03', '06-30', '08-09', '09-20', '11-01', '12-13'],
    2006: ['01-31', '03-28', '05-10', '06-29', '08-08', '09-20', '10-25', '12-12'],
    2007: ['01-31', '03-21', '05-09', '06-28', '08-07', '09-18', '10-31', '12-11'],
    2008: ['01-30', '03-18', '04-30', '06-25', '08-05', '09-16', '10-29', '12-16'],
    2009: ['01-28', '03-18', '04-29', '06-24', '08-12', '09-23', '11-04', '12-16'],
    2010: ['01-27', '03-16', '04-28', '06-23', '08-10', '09-21', '11-03', '12-14'],
    2011: ['01-26', '03-15', '04-27', '06-22', '08-09', '09-21', '11-02', '12-13'],
    2012: ['01-25', '03-13', '04-25', '06-20', '08-01', '09-13', '10-24', '12-12'],
    2013: ['01-30', '03-20', '05-01', '06-19', '07-31', '09-18', '10-30', '12-18'],
    2014: ['01-29', '03-19', '04-30', '06-18', '07-30', '09-17', '10-29', '12-17'],
    2015: ['01-28', '03-18', '04-29', '06-17', '07-29', '09-17', '10-28', '12-16'],
    2016: ['01-27', '03-16', '04-27', '06-15', '07-27', '09-21', '11-02', '12-14'],
    2017: ['02-01', '03-15', '05-03', '06-14', '07-26', '09-20', '11-01', '12-13'],
    2018: ['01-31', '03-21', '05-02', '06-13', '08-01', '09-26', '11-08', '12-19'],
    2019: ['01-30', '03-20', '05-01', '06-19', '07-31', '09-18', '10-30', '12-11'],
    2020: ['01-29', '03-18', '04-29', '06-10', '07-29', '09-16', '11-05', '12-16'],
    2021: ['01-27', '03-17', '04-28', '06-16', '07-28', '09-22', '11-03', '12-15'],
    2022: ['01-26', '03-16', '05-04', '06-15', '07-27', '09-21', '11-02', '12-14'],
    2023: ['02-01', '03-22', '05-03', '06-14', '07-26', '09-20', '11-01', '12-13'],
    2024: ['01-31', '03-20', '05-01', '06-12', '07-31', '09-18', '11-07', '12-18'],
    2025: ['01-29', '03-19', '05-07', '06-18', '07-30', '09-17', '10-29', '12-10'],
    2026: ['01-28', '03-18', '04-29', '06-17', '07-29', '09-16', '10-28', '12-09'],
    2027: ['01-27', '03-17', '04-28', '06-09', '07-28', '09-15', '10-27', '12-08'],
}
FOMC = pd.DatetimeIndex(sorted(pd.Timestamp(f'{y}-{d}') for y, ds in _F.items() for d in ds))
HOL = USFederalHolidayCalendar().holidays(start='1999-12-01', end='2028-12-31')
BD = pd.bdate_range('1999-01-01', '2028-12-31', freq='C', holidays=list(HOL.values.astype('datetime64[D]')))
NB = len(BD)

_BOJ = """2013-01-22 2013-02-07 2013-03-07 2013-04-04 2013-05-22 2013-06-11 2013-07-11 2013-08-08 2013-09-05 2013-10-04 2013-11-21 2013-12-20
2014-01-22 2014-02-18 2014-03-11 2014-04-08 2014-05-21 2014-06-13 2014-07-15 2014-08-08 2014-09-04 2014-10-07 2014-10-31 2014-11-19 2014-12-19
2015-01-21 2015-02-18 2015-03-17 2015-04-08 2015-04-30 2015-05-22 2015-06-19 2015-07-15 2015-08-07 2015-09-15 2015-10-07 2015-10-30 2015-11-19 2015-12-18
2016-01-29 2016-03-15 2016-04-28 2016-06-16 2016-07-29 2016-09-21 2016-11-01 2016-12-20
2017-01-31 2017-03-16 2017-04-27 2017-06-16 2017-07-20 2017-09-21 2017-10-31 2017-12-21
2018-01-23 2018-03-09 2018-04-27 2018-06-15 2018-07-31 2018-09-19 2018-10-31 2018-12-20
2019-01-23 2019-03-15 2019-04-25 2019-06-20 2019-07-30 2019-09-19 2019-10-31 2019-12-19
2020-01-21 2020-03-16 2020-04-27 2020-06-16 2020-07-15 2020-09-17 2020-10-29 2020-12-18
2021-01-21 2021-03-19 2021-04-27 2021-06-18 2021-07-16 2021-09-22 2021-10-28 2021-12-17
2022-01-18 2022-03-18 2022-04-28 2022-06-17 2022-07-21 2022-09-22 2022-10-28 2022-12-20
2023-01-18 2023-03-10 2023-04-28 2023-06-16 2023-07-28 2023-09-22 2023-10-31 2023-12-19
2024-01-23 2024-03-19 2024-04-26 2024-06-14 2024-07-31 2024-09-20 2024-10-31 2024-12-19
2025-01-24 2025-03-19 2025-05-01 2025-06-17 2025-07-31 2025-09-19 2025-10-30 2025-12-19
2026-01-23 2026-03-19 2026-04-28 2026-06-16 2026-07-31 2026-09-18 2026-10-30 2026-12-18""".split()
_ECB = """2015-01-22 2015-03-05 2015-04-15 2015-06-03 2015-07-16 2015-09-03 2015-10-22 2015-12-03
2016-01-21 2016-03-10 2016-04-21 2016-06-02 2016-07-21 2016-09-08 2016-10-20 2016-12-08
2017-01-19 2017-03-09 2017-04-27 2017-06-08 2017-07-20 2017-09-07 2017-10-26 2017-12-14
2018-01-25 2018-03-08 2018-04-26 2018-06-14 2018-07-26 2018-09-13 2018-10-25 2018-12-13
2019-01-24 2019-03-07 2019-04-10 2019-06-06 2019-07-25 2019-09-12 2019-10-24 2019-12-12
2020-01-23 2020-03-12 2020-04-30 2020-06-04 2020-07-16 2020-09-10 2020-10-29 2020-12-10
2021-01-21 2021-03-11 2021-04-22 2021-06-10 2021-07-22 2021-09-09 2021-10-28 2021-12-16
2022-02-03 2022-03-10 2022-04-14 2022-06-09 2022-07-21 2022-09-08 2022-10-27 2022-12-15
2023-02-02 2023-03-16 2023-05-04 2023-06-15 2023-07-27 2023-09-14 2023-10-26 2023-12-14
2024-01-25 2024-03-07 2024-04-11 2024-06-06 2024-07-18 2024-09-12 2024-10-17 2024-12-12
2025-01-30 2025-03-06 2025-04-17 2025-06-05 2025-07-24 2025-09-11 2025-10-30 2025-12-18
2026-02-05 2026-03-19 2026-04-30 2026-06-11 2026-07-23 2026-09-10 2026-10-29 2026-12-17""".split()
BOJ = pd.DatetimeIndex(pd.to_datetime(_BOJ))
ECB = pd.DatetimeIndex(pd.to_datetime(_ECB))


def _nfp():
    """BLS Employment Situation: third Friday after the Saturday ending the reference week (the week containing the 12th);
    +7 days if that Friday is Jan 1, -1 day for other federal holidays (July 4 -> Thursday)."""
    hol = set(HOL)
    out = []
    for y in range(1999, 2028):
        for m in range(1, 13):
            d12 = pd.Timestamp(y, m, 12)
            sat = d12 + pd.Timedelta(days=(5 - d12.weekday()) % 7)
            rel = sat + pd.Timedelta(days=20)
            if rel in hol:
                rel = rel + pd.Timedelta(days=7) if rel.month == 1 else rel - pd.Timedelta(days=1)
            out.append(rel)
    return pd.DatetimeIndex(out)
NFP = _nfp()

# ---------------------------------------------------------------- parameters (variance ratios of a day vs a normal day)
UST = [('F', (-2, -1), 0.63), ('F', (0,), 1.1), ('F', (1,), 1.9), ('F', (2,), 0.9), ('N', (0,), 2.2)]
FX = [('F', (-2, -1, 0), 0.76), ('F', (1,), 2.4), ('F', (2,), 1.2), ('N', (0,), 1.35)]
PARAMS = {'UST': UST, 'FX': FX,
          'EUR': [('E', (0,), 2.5)], 'DKK': [('E', (0,), 2.5)], 'CHF': [('E', (0,), 1.6)], 'SEK': [('E', (0,), 1.5)], 'NOK': [('E', (0,), 1.5)],
          'JPY': [('B', (0,), 2.9)], 'AUD': [('B', (0,), 1.25)], 'NZD': [('B', (0,), 1.25)], 'CAD': [('B', (0,), 1.25)]}
GATE = {'B': pd.Timestamp('2014-06-01'), 'E': pd.Timestamp('2016-04-01')}   # calendar must cover the trailing 300 BD
# swarm3/combined 'full': UST (all cards) and ECB-fixed FX tables.  FS / FO = FOMC in an SEP month (Mar/Jun/Sep/Dec) / other month,
# EP / EO = ECB meeting in a projection month / other month (gated like E).  H.10 (non-ECB) FX keeps FX + PARAMS above.
UST_NOTE = [('F', (-2, -1), 0.78), ('FS', (0,), 1.84), ('FO', (0,), 1.4), ('FS', (1,), 1.73), ('FO', (1,), 1.28), ('FS', (2,), 1.08), ('FO', (2,), 1.18),
            ('N', (0,), 2.31)]
UST_BILL = [('F', (-2, -1), 0.82), ('FS', (0,), 2.57), ('FO', (0,), 1.61), ('FS', (1,), 1.52), ('FO', (1,), 1.29), ('FS', (2,), 0.99), ('FO', (2,), 0.92),
            ('N', (0,), 2.16)]
EM_IDS = ('MXN', 'BRL', 'ZAR', 'KRW', 'INR')
FX_IDS = ('AUD', 'CAD', 'CHF', 'DKK', 'EUR', 'GBP', 'JPY', 'NOK', 'NZD', 'SEK') + EM_IDS
EM_ECB = [('F', (-2, -1, 0), 0.95), ('FS', (1,), 2.44), ('FO', (1,), 1.64), ('FS', (2,), 1.28), ('FO', (2,), 1.25), ('N', (0,), 1.57)]   # payrolls +1 BD
ECB_FX = {'EUR': [('EP', (0,), 2.2), ('EO', (0,), 1.4), ('B', (0,), 1.27)], 'DKK': [('EP', (0,), 2.2), ('EO', (0,), 1.4), ('B', (0,), 1.27)],
          'CHF': [('EP', (0,), 1.8), ('EO', (0,), 1.4), ('B', (0,), 1.6)], 'SEK': [('EP', (0,), 1.15), ('EO', (0,), 1.05), ('B', (0,), 1.2)],
          'NOK': [('EP', (0,), 1.15), ('EO', (0,), 1.05), ('B', (0,), 1.2)], 'GBP': [('EP', (0,), 2.1), ('B', (0,), 1.4)],
          'JPY': [('EP', (0,), 1.35), ('B', (0,), 2.9)], 'AUD': [('B', (0,), 1.25)], 'NZD': [('B', (0,), 1.25)]}
_SEPF = np.isin(FOMC.month, [3, 6, 9, 12]); _PRJE = np.isin(ECB.month, [3, 6, 9, 12])
_DATES = {'F': FOMC, 'N': NFP, 'B': BOJ, 'E': ECB, 'FS': FOMC[_SEPF], 'FO': FOMC[~_SEPF], 'EP': ECB[_PRJE], 'EO': ECB[~_PRJE]}
_MARK, _SER = {}, {}


def _mark(kind, off):
    k = (kind, off)
    if k not in _MARK:
        v = np.zeros(NB, bool)
        pos = BD.searchsorted(_DATES[kind]) + off
        pos = pos[(pos >= 0) & (pos < NB)]
        v[pos] = True
        _MARK[k] = v
    return _MARK[k]


def _series(asset, asof, ecb_fix=False):
    u = str(asset).upper()
    em = u in EM_IDS
    if u.startswith('UST'):
        tab = UST_BILL if is_bill(asset) else UST_NOTE
    elif u not in FX_IDS:
        tab = []                 # unrecognised id (generic short-horizon path): no event timing
    elif ecb_fix:
        tab = (EM_ECB if em else FX) + ECB_FX.get(u, [])
    else:
        tab = FX + PARAMS.get(u, [])
    on = tuple(sorted(k for k in GATE if pd.Timestamp(asof) >= GATE[k]))
    fp = _final_prior(asset, asof)
    key = (asset, on, ecb_fix, fp)
    if key not in _SER:
        m = np.ones(NB)
        for kind, offs, r in tab:
            if ('E' if kind in ('EP', 'EO') else kind) in GATE and ('E' if kind in ('EP', 'EO') else kind) not in on:
                continue
            for o in offs:
                if ecb_fix and kind in ('E', 'EP', 'EO'):
                    o = o + 1        # ECB reference rates are fixed at 14:10 CET; the ECB decision comes at 14:15 CET, so it shows the next day
                if ecb_fix and em and kind == 'N':
                    o = o + 1        # payrolls (8:30 ET) come after the 14:10 CET fix in most weeks
                m = m + _mark(kind, o) * (r - 1.0)          # independent shocks: variances add
        if fp is not None:
            pos = BD.searchsorted(FINAL_FOMC_PRIOR['decision'])
            if 0 <= pos < NB and BD[pos] == FINAL_FOMC_PRIOR['decision']:
                m[pos] = m[pos] * fp * fp                   # sd x fp on the decision-day increment only
        _SER[key] = np.maximum(m, 0.25)
    return _SER[key]


def _cum(asof, hs, m, ntr=300):
    pos = BD.searchsorted(pd.Timestamp(asof))
    T = m[pos - ntr + 1: pos + 1].mean()
    cs = np.concatenate([[0.0], np.cumsum(m[pos + 1: pos + 1 + int(max(hs))])])
    return np.array([cs[int(h)] for h in hs]), T


def apply_events(card, dev, gamma=1.0, tail_keep=0.0, z0=1.0, z1=2.4, ntr=300):
    """Remap a deviation matrix (500, n_cells; columns in card['cells'] order, deviations from the centre in final units) by the
    per-asset event-timed increments (ntr = number of trailing business days the base volatility was estimated on, 300 for M0).  tail_keep > 0: where a draw was shrunk, add back tail_keep * ramp(|z|) of the removed part."""
    dev = np.asarray(dev, float)
    out = dev.copy()
    by = {}
    for j, (a, h) in enumerate(card['cells']):
        by.setdefault(a, []).append((int(h), j))
    for a, lst in by.items():
        lst.sort()
        m = _series(a, card['asof'])
        hs = [h for h, _ in lst]
        S, T = _cum(card['asof'], hs, m, ntr)
        prevS, prevh = 0.0, 0
        acc = np.zeros(dev.shape[0]); prevcol = np.zeros(dev.shape[0])
        for (h, j), s in zip(lst, S):
            if h == prevh:
                out[:, j] = acc
                continue
            f = ((s - prevS) / (h - prevh) / T) ** (0.5 * gamma)
            acc = acc + f * (dev[:, j] - prevcol)
            out[:, j] = acc
            prevcol, prevS, prevh = dev[:, j], s, h
    if tail_keep > 0:
        sd = dev.std(0, keepdims=True)
        w = np.clip((np.abs(dev) / sd - z0) / (z1 - z0), 0, 1) * tail_keep
        out = np.where(np.abs(out) < np.abs(dev), out + w * (dev - out), out)
    return out




N_DRAWS = 500
LAM, POW, LO, HI = 0.94, 0.8, 0.5, 2.0
FX_LAM = 0.25
NOTE_LAM = 0.5           # UST notes centre = anchor + NOTE_LAM * (v12 centre - anchor); bills keep the v12 centre
BILL_CLIP = (0.35, 3.0)  # EWMA scale clip for bills
# One-off decision-uncertainty prior for the ranked Final's FOMC (statement Wed 28 Oct 2026; ranked as-of Mon 26 Oct 2026).
# Market-implied odds of a 25 bp move on 25 Oct 2026 (CME FedWatch / prediction markets) are far from 0 and 1, so the bill
# increments on the decision day carry a two-sided jump that the history-fitted FO(0) ratio (priced meetings) does not cover.
# sd multiplier on the decision-day increment per bill tenor; variance = sd^2. Off outside the as-of window (public, practice
# and check-run cards are unchanged). Disclosed in ARTIFACT_PROVENANCE.md. Set/confirm the values on 24-25 Oct from the odds then.
FINAL_FOMC_PRIOR = {'asof_lo': pd.Timestamp('2026-10-23'), 'asof_hi': pd.Timestamp('2026-10-27'), 'decision': pd.Timestamp('2026-10-28'),
                    'sd': {'3M': 2.5, '6M': 2.0, '1Y': 1.5}}


def _final_prior(asset, asof):
    pr = FINAL_FOMC_PRIOR
    if pr is None or not (pr['asof_lo'] <= pd.Timestamp(asof) <= pr['asof_hi']) or not is_bill(asset):
        return None
    t = str(asset).upper().split('_')[-1]
    for k, v in pr['sd'].items():
        if t.endswith(k) or t == k:
            return float(v)
    return None
GRID_DF = 12             # variance-preserving Student-t(GRID_DF) quantile grid (even df; None = Gaussian)
COPULA_NU = 2            # t-copula: M0 ranks taken from W / sqrt(chi2_nu / nu), one draw per M0 draw shared by the card's cells (None = off)
D1_ENABLED = True        # swarm4/fresh D1, adopted by Luke 2026-10-14: D1_DRAWS draws, eigen-ordered Halton copula, fair-CRPS-optimal grid positions
D1_DRAWS = 200           # the scorer's minimum draw count; a larger card n_draws_min wins
PRIMES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97)
_GRID = {}


def _ygrid(n):
    if n not in _GRID:
        inv = NormalDist().inv_cdf
        y = np.array([inv((k + 0.5) / n) for k in range(n)])
        y = y - y.mean()
        _GRID[n] = y / y.std()
    return _GRID[n]


def is_bill(a):
    t = str(a).upper().split('_')[-1]
    return (('M' in t or 'W' in t) and 'Y' not in t) or t == '1Y'


def _t_cdf_even(t, df):
    th = np.arctan(t / np.sqrt(df)); c2 = np.cos(th) ** 2
    term, tot = 1.0, 1.0
    for k in range(1, df // 2):
        term = term * c2 * (2 * k - 1) / (2 * k)
        tot = tot + term
    return 0.5 + 0.5 * np.sign(t) * np.sin(np.abs(th)) * tot


def _t_ppf_even(p, df):
    lo = np.full_like(p, -60.0); hi = np.full_like(p, 60.0)
    for _ in range(100):
        mid = 0.5 * (lo + hi); up = _t_cdf_even(mid, df) < p
        lo = np.where(up, mid, lo); hi = np.where(up, hi, mid)
    return 0.5 * (lo + hi)


def _tgrid(n, df):
    key = (n, df)
    if key not in _GRID:
        x = _t_ppf_even((np.arange(n) + 0.5) / n, df); x = x - x.mean()
        _GRID[key] = x / x.std()
    return _GRID[key]


def _fair_grid(m, df):
    """m-point unit-sd grid at the fair-CRPS-optimal positions F = i/(m-1), end points clipped at 0.25/m; Student-t(df), or Gaussian for None"""
    key = ('fair', m, df)
    if key not in _GRID:
        p = np.clip(np.arange(m) / (m - 1), 0.25 / m, 1 - 0.25 / m)
        x = _t_ppf_even(p, df) if df else np.array([NormalDist().inv_cdf(q) for q in p])
        x = x - x.mean()
        _GRID[key] = x / x.std()
    return _GRID[key]


def _halton(m, d, skip=20):
    key = ('halton', m, d)
    if key not in _GRID:
        out = np.empty((m, d))
        for j in range(d):
            b = PRIMES[j]
            for i in range(m):
                k, f, r = i + skip, 1.0, 0.0
                while k > 0:
                    f /= b; r += f * (k % b); k //= b
                out[i, j] = r
        _GRID[key] = out
    return _GRID[key]


def _d1_copula(covT, sd, m):
    """(m, n) D1 copula sample: Halton points (prime bases ascending) on the eigen-directions of the model correlation in descending
    eigenvalue order, antithetic covariance-matched normals (fixed seed) past the prime list; with COPULA_NU the t-copula scale takes
    the next Halton dimension (closed form for nu = 2: chi2_2 / 2 = -ln(1 - u))."""
    n = len(sd)
    sdd = np.where(sd > 0, sd, 1); R = covT / np.outer(sdd, sdd)
    w, V = np.linalg.eigh((R + R.T) / 2); o = np.argsort(w)[::-1]
    L = V[:, o] * np.sqrt(np.clip(w[o], 0, None))
    t = 1 if COPULA_NU and n > 1 else 0
    if t and COPULA_NU != 2:
        raise ValueError('D1 t-copula needs COPULA_NU = 2')
    k = min(n, len(PRIMES) - t)
    H = _halton(m, k + t)
    inv = NormalDist().inv_cdf
    Z = np.array([[inv(u) for u in row] for row in H[:, :k]])
    if k < n:
        z = np.random.default_rng(0).standard_normal(((m + 1) // 2, n - k)); z = np.vstack([z, -z])[:m]
        w2, V2 = np.linalg.eigh(z.T @ z / m)
        Z = np.hstack([Z, z @ (V2 / np.sqrt(np.clip(w2, 1e-12, None))) @ V2.T])
    W = Z @ L.T
    return W / np.sqrt(-np.log(1 - H[:, k]))[:, None] if t else W


def ew_scale(cells, hist):
    out = np.ones(len(cells)); cache = {}
    for j, (a, h) in enumerate(cells):
        if a not in cache:
            dx = np.diff(np.asarray(hist[a], float)[-1500:])
            s300 = dx[-300:].std(ddof=1)
            x2 = dx[-1000:] ** 2
            init = np.mean(dx[-300:] ** 2) if len(dx) >= 300 else x2.mean()
            vn = init
            for x in x2:                       # EWMA recursion v_t = (1 - LAM) x_t + LAM v_(t-1), started at init
                vn = (1 - LAM) * x + LAM * vn
            cache[a] = (s300, np.sqrt(vn))
        s300, ewn = cache[a]
        out[j] = np.clip((ewn / s300) ** POW, *(BILL_CLIP if a.upper().startswith('UST') and is_bill(a) else (LO, HI))) if s300 > 0 else 1.0
    return out


def event_map(cells, asof, ntr=300, ecb_fix=False):
    """Linear map M (n x n) on the cells: deviations @ M rescales each asset's increments between consecutive horizons by
    sqrt(event-adjusted variance / trailing-average variance)."""
    n = len(cells)
    M = np.zeros((n, n))
    by = {}
    for j, (a, h) in enumerate(cells):
        by.setdefault(a, []).append((int(h), j))
    for a, lst in by.items():
        lst.sort()
        m = _series(a, asof, ecb_fix)
        hs = [h for h, _ in lst]
        S, T = _cum(asof, hs, m, ntr)
        prevS, prevh, prevj = 0.0, 0, None
        acc = np.zeros(n)
        for (h, j), s in zip(lst, S):
            if h == prevh:
                M[:, j] = acc
                continue
            f = ((s - prevS) / (h - prevh) / T) ** 0.5
            inc = np.zeros(n); inc[j] = 1.0
            if prevj is not None:
                inc[prevj] -= 1.0
            acc = acc + f * inc
            M[:, j] = acc
            prevS, prevh, prevj = s, h, j
    return M


def draws(cells, anchor, v12c, base, hist, Sig, steps, assets, asof, ecb_fix=False, unit=None, n_min=0, rec=None):
    """cells: [(asset, h)] sorted; anchor, v12c: per-cell arrays; base: (n_draws, n) M0 deviations; hist: {asset: levels up to asof};
    Sig: M0 daily-change covariance (assets order); steps: per-cell step counts used by M0; asof: Timestamp."""
    asof = pd.Timestamp(asof)
    n = len(cells)
    ust = np.array([a.upper().startswith('UST') for a, _ in cells])
    anchor = np.asarray(anchor, float); v12c = np.asarray(v12c, float); base = np.asarray(base, float)
    note = np.array([a.upper().startswith('UST') and not is_bill(a) for a, _ in cells])
    centre = np.where(ust, np.where(note, anchor + NOTE_LAM * (v12c - anchor), v12c), anchor + FX_LAM * (v12c - anchor))
    s = ew_scale(cells, hist)
    M = event_map(cells, asof, ecb_fix=ecb_fix)
    W = (base * s[None, :]) @ M
    idx = {a: i for i, a in enumerate(assets)}
    cov = np.array([[min(steps[p], steps[q]) * Sig[idx[cells[p][0]], idx[cells[q][0]]] for q in range(n)] for p in range(n)])
    covT = M.T @ (cov * np.outer(s, s)) @ M
    sd = np.sqrt(np.clip(np.diag(covT), 0, None))
    if D1_ENABLED:
        m = max(D1_DRAWS, int(n_min))
        W = _d1_copula(covT, sd, m)
        g = _fair_grid(m, int(GRID_DF) if GRID_DF else None)
        if rec is not None:
            rec['d1'] = True
    else:
        if COPULA_NU and n > 1 and unit is not None:
            rng = np.random.default_rng((zlib.crc32(str(unit).encode()) & 0x7FFFFFFF) ^ 12345)
            W = W / np.sqrt(rng.chisquare(COPULA_NU, size=(W.shape[0], 1)) / COPULA_NU)
        g = _tgrid(W.shape[0], int(GRID_DF)) if GRID_DF else _ygrid(W.shape[0])
    order = np.argsort(W, axis=0)
    out = np.empty_like(W)
    out[order, np.arange(n)[None, :]] = g[:, None] * sd[None, :]
    return centre[None, :] + out
