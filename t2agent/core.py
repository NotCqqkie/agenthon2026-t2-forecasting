import json, pathlib, re, tomllib, zlib
from . import v9 as V9
from . import v10 as V10
from . import v11 as V11
import numpy as np
import pandas as pd

N_DRAWS = 500
MIN_DRAWS = 200          # the scorer's limits.min_draws; finite_or_fallback keeps a finite output of at least this many draws (D1 writes 200)
WINDOW = 300
RETURN_TYPES = frozenset({'log_return', 'return', 'simple_return', 'pct_change'})


def norm_ttype(v):
    return 'log_return' if str(v).strip().lower() in RETURN_TYPES else 'level'

PARAMS = dict(k=0.5, c=0.7, ct=1.2)
FAMILY_PARAMS = {'F1': dict(k=0.625, c=0.5, ct=1.0, ct2=1.15), 'F2': dict(k=0.5, c=1.0, ct=1.3),
                 'F3': dict(k=0.375, c=0.9, ct=1.3), 'F4': dict(k=0.625, c=1.6, ct=1.7, beta=0.5)}
VOL_BETA, VOL_CLIP = 0.25, (0.7, 1.4)
RISK_OFF = {'MKT': -1, 'SMB': -1, 'QMJ': 1, 'AUD': -1, 'NZD': -1, 'EUR': -1, 'GBP': -1, 'CAD': 1, 'NOK': 1,
            'SEK': 1, 'DKK': 1, 'JPY': -1, 'CHF': -1}
F4_SKEW = (0.3, 0.25)
# v8 additions (2026-10-02 swarm, verified on a sealed holdout):
# A: Treasury prior-window reversal of the centre drift
REV_G0 = {'F1': 0.75, 'F2': 1.0, 'F3': 1.0, 'F4': 1.25}
REV_PW, REV_C3, REV_CAP = 0.5, 0.5, 1.5
# C: drift-direction centre shift linear in the as-of vol ratio; F4 inner width by trend strength
REGIME_RULE = {'F1': ('lvm', 0.049, 0.0645, 0.1, 0.0), 'F2': ('lvm', -0.026, 0.178, 0.1, 0.0),
               'F3': ('lvm', 0.094, 0.0, 0.1, 0.0), 'F4': ('lv60', 0.0608, 0.3735, 0.1, 0.5)}
REGIME_NORM = {'lv20': (-0.115, 0.33), 'lv60': (-0.06, 0.25), 'lvm': (-0.09, 0.28), 'lt60': (-0.22, 0.68)}
TREND_WIDTH = {'F4': 0.106}
# B: F1 one-asset two-horizon construction; F3 log-return dispersion match
PAIR_LAM = 0.5
TAU_GRID = np.exp(np.linspace(np.log(0.4), np.log(2.5), 25))


def find_card(panels_dir):
    for cand in (panels_dir / 'card.toml', panels_dir.parent / 'card.toml'):
        if cand.exists():
            return cand
    raise FileNotFoundError('card.toml')


def read_panels(panels_dir):
    found = sorted(panels_dir.glob('*.parquet'))
    if not found:
        found = sorted(panels_dir.parent.glob('*.parquet'))
    if not found:
        found = sorted(panels_dir.glob('panels/*.parquet'))
    out = []
    for p in found:
        try:
            df = pd.read_parquet(p)
        except Exception:
            continue
        col = 'asset' if 'asset' in df.columns else ('asset_id' if 'asset_id' in df.columns else None)
        if col is None or 'date' not in df.columns or 'value' not in df.columns:
            continue
        df = df[['date', col, 'value']].rename(columns={col: 'asset'})
        df['asset'] = df['asset'].astype(str)
        if not pd.api.types.is_numeric_dtype(df['value']):
            df['value'] = pd.to_numeric(df['value'], errors='coerce')
        df['date'] = pd.to_datetime(df['date'].astype(str).str.slice(0, 10), errors='coerce')
        out.append((p.name, df))
    return out


def add_asof_levels(panels, levels, asof):
    """spec as_of_level for an asset whose newest row (<= as-of) is more than 30 days old: append the level as an as-of row"""
    out = []
    for name, df in panels:
        add = []
        for a, lv in levels.items():
            sub = df[df['asset'] == str(a)]
            if len(sub):
                last = sub['date'][sub['date'] <= asof].max()
                try:
                    v = float(lv)
                except Exception:
                    continue
                if np.isfinite(v) and (pd.isna(last) or (asof - last).days > 30):
                    add.append({'date': asof, 'asset': str(a), 'value': v})
        out.append((name, pd.concat([df, pd.DataFrame(add)], ignore_index=True) if add else df))
    return out


def _norm_id(x):
    return str(x).strip().casefold()


def _alnum_id(x):
    return re.sub(r'[-_ /]', '', str(x).casefold())


def asset_rows(panels, asset):
    """(panel name, rows) for an asset id: exact match in the first panel holding it; only if no panel has an exact match,
    the first panel whose ids match after strip + casefold (hand-typed ids with stray case/whitespace); then the alias_id form
    (DGS3MO -> UST_3M, USDMXN -> MXN); last, a unique id that matches once '_', '-', '/' and spaces are removed (UST-3M / ust3m)"""
    for name, df in panels:
        sub = df[df['asset'] == asset]
        if len(sub):
            return name, sub
    key = _norm_id(asset)
    for name, df in panels:
        sub = df[df['asset'].map(_norm_id) == key]
        if len(sub):
            return name, sub
    ca = alias_id(asset)
    if ca != asset:
        for name, df in panels:
            sub = df[df['asset'] == ca]
            if len(sub):
                return name, sub
    key = _alnum_id(asset)
    if key:
        for name, df in panels:
            sub = df[df['asset'].map(_alnum_id) == key]
            if len(sub) and sub['asset'].nunique() == 1:
                return name, sub
    return None, df.iloc[0:0] if panels else pd.DataFrame(columns=['date', 'asset', 'value'])


def history(panels, asset, asof):
    if panels:
        _, sub = asset_rows(panels, asset)
        if len(sub):
            sub = sub[sub['date'] <= asof].dropna(subset=['date']).sort_values('date', kind='stable')
            sub = sub.drop_duplicates('date', keep='last')
            sub = sub[np.isfinite(pd.to_numeric(sub['value'], errors='coerce'))]
            return pd.Series(sub['value'].astype(float).to_numpy(), index=pd.DatetimeIndex(sub['date']))
    raise KeyError(f'asset {asset} not in panels')


def steps_of(s, ttype):
    s = s.iloc[-WINDOW:]
    if ttype == 'log_return' and np.median(np.abs(s.to_numpy())) >= 0.2:
        raise ValueError('log_return panel does not look like returns')
    gaps = s.index.to_series().diff().dt.days
    hole = max(10 * float(gaps.iloc[1:].median()), 5.0) if len(s) > 1 else 5.0
    st = s.diff() if ttype == 'level' else s.copy()
    st[gaps > hole] = np.nan
    return st.iloc[1:], s, hole


def panel_steps(s_win, hole, h, target_date, asof, named=False):
    if named and target_date is not None and len(s_win):
        last = s_win.index[-1]
        steps = 12 * (target_date.year - last.year) + (target_date.month - last.month)
        return steps if steps > 0 else h
    if len(s_win) < 3:
        return h
    gaps = s_win.index.to_series().diff().dt.days.iloc[1:]
    gaps = gaps[gaps <= hole]
    spacing = float(gaps.mean()) if len(gaps) else 1.0
    if target_date is None:
        if spacing <= 20:
            return h
        target_date = asof + pd.offsets.BDay(h)
    last = s_win.index[-1]
    if spacing > 20:
        steps = 12 * (target_date.year - last.year) + (target_date.month - last.month)
    else:
        steps = int(round((target_date - last).days / spacing))
    if steps <= 0:
        return h
    ratio = max(steps, h) / max(min(steps, h), 1)
    return steps if ratio >= 2 else h


def target_dates(card, spec, asof, cells):
    """Per-cell (target date, named month): observation_periods / questions[].observation_period name the month (M0 §3.7);
    target_dates / questions[].target_date are dates for steps 1-5."""
    out, named = {}, {}
    t = card.get('targets', {})
    st = spec.get('targets', {}) if isinstance(spec, dict) else {}
    st = st if isinstance(st, dict) else {}
    hs = list(t.get('horizons', []))
    for src in (t, st):
        for key in ('target_dates', 'observation_periods'):
            vals = src.get(key)
            if isinstance(vals, list) and len(vals) == len(hs):
                for h, v in zip(hs, vals):
                    try:
                        ts = pd.Timestamp(str(v)[:10] if key == 'target_dates' else str(v)[:7] + '-01')
                    except Exception:
                        continue
                    for c in cells:
                        if c[1] == int(h):
                            if key == 'observation_periods':
                                named[c] = ts
                            else:
                                out.setdefault(c, ts)
    for src in (card, spec):
        qs = src.get('questions', []) if isinstance(src, dict) else []
        for row in qs if isinstance(qs, list) else []:
            if not isinstance(row, dict):
                continue
            try:
                c = (str(row.get('asset')), int(row.get('horizon')))
            except Exception:
                continue
            if c not in cells:
                continue
            try:
                if 'observation_period' in row:
                    named[c] = pd.Timestamp(str(row['observation_period'])[:7] + '-01')
                elif 'target_date' in row:
                    out.setdefault(c, pd.Timestamp(str(row['target_date'])[:10]))
            except Exception:
                pass
    return {c: (named[c], True) if c in named else (out.get(c), False) for c in cells}


def m0_parts(card, spec, panels, asof, unit_id):
    t = card['targets']
    ttype = norm_ttype(t.get('target_type', 'level'))
    assets = sorted(str(a) for a in t['asset_ids'])
    cells = sorted((str(a), int(h)) for a in t['asset_ids'] for h in t['horizons'])
    hist, steps, wins, holes = {}, {}, {}, {}
    for a in assets:
        hist[a] = history(panels, a, asof)
        steps[a], wins[a], holes[a] = steps_of(hist[a], ttype)
    frame = pd.DataFrame(steps).dropna()
    if len(assets) > 1 and len(frame) < 10:
        own = [steps[a].dropna().to_numpy() for a in assets]
        mu = np.array([x.mean() if len(x) else np.nan for x in own])
        Sig = np.diag([x.var(ddof=1) if len(x) > 1 else np.nan for x in own])
    else:
        X = frame[assets].to_numpy()
        mu = X.mean(0)
        Sig = np.atleast_2d(np.cov(X, rowvar=False))
    idx = {a: i for i, a in enumerate(assets)}
    tds = target_dates(card, spec, asof, cells)
    s = {c: panel_steps(wins[c[0]], holes[c[0]], c[1], tds[c][0], asof, tds[c][1]) for c in cells}
    anchor = np.array([hist[a].iloc[-1] if ttype == 'level' else 0.0 for a, _ in cells])
    drift = np.array([s[c] * mu[idx[c[0]]] for c in cells])
    cov = np.array([[min(s[c1], s[c2]) * Sig[idx[c1[0]], idx[c2[0]]] for c2 in cells] for c1 in cells])
    cov[np.diag_indices_from(cov)] += 1e-10
    eye = 1e-9 * np.eye(len(cells))
    try:
        L = np.linalg.cholesky(cov + eye)
    except np.linalg.LinAlgError:
        L = np.diag(np.sqrt(np.diag(cov + eye)))
    rng = np.random.default_rng(zlib.crc32(unit_id.encode()) & 0x7FFFFFFF)
    Z = rng.standard_normal((N_DRAWS, len(cells)))
    return dict(cells=cells, anchor=anchor, drift=drift, base=Z @ L.T, ttype=ttype, steps=s,
                mu=mu, Sig=Sig, assets=assets, n_rows=len(frame), hist=hist)


def vol_ratio(p, beta=None):
    out = []
    for a, _ in p['cells']:
        s = p['hist'][a].to_numpy()
        st = np.diff(s) if p['ttype'] == 'level' else s
        st = st[np.isfinite(st)]
        v20, v300 = (st[-20:].std(), st[-300:].std()) if len(st) >= 40 else (1.0, 1.0)
        out.append(v20 / v300 if v300 > 0 and np.isfinite(v20 / v300) else 1.0)
    return np.clip(np.array(out) ** (VOL_BETA if beta is None else beta), *VOL_CLIP)


def transform(p, k, c, ct, vol=False, adj=None, skew=None, ct2=None, beta=None):
    base = p['base']
    sd = base.std(0, keepdims=True)
    sd = np.where(sd > 0, sd, 1.0)
    z = base / sd
    a = np.abs(z)
    ro = np.array([RISK_OFF.get(str(x).upper(), 0) for x, _ in p['cells']], float)[None, :]
    m = (1 + skew[0] * np.sign(z) * ro) if skew else 1.0
    ct2 = ct if ct2 is None else ct2
    zt = np.sign(z) * (c * np.minimum(a, 1) + ct * m * np.clip(a - 1, 0, 1) + ct2 * m * np.maximum(a - 2, 0))
    if vol:
        zt = zt * vol_ratio(p, beta)[None, :]
    if skew:
        zt = zt + skew[1] * ro
    center = p['anchor'] + k * p['drift']
    if adj is not None:
        shift, width = adj
        zt = zt * width[None, :]
        center = center + shift
    X = center + sd * zt
    if X.shape[1] == 2:
        m0 = p['anchor'] + p['drift'] + base
        X = m0 + (X[:, :1] - m0[:, :1])
    return X


def cell_feats(p):
    out = []
    for a, _ in p['cells']:
        x = p['hist'][a].to_numpy()
        st = np.diff(x) if p['ttype'] == 'level' else x
        if len(st) < 61:
            out.append(None)
            continue
        out.append(dict(v20=st[-20:].std(), v60=st[-60:].std(), v300=st[-300:].std(),
                        r60=(x[-1] - x[-61]) if p['ttype'] == 'level' else st[-60:].sum()))
    return out


def reversal_drift(p, fam, k, sd):
    d = np.array(p['drift'], float).copy()
    if p['ttype'] != 'level' or fam not in REV_G0:
        return k * d
    for j, (a, h) in enumerate(p['cells']):
        if not str(a).upper().startswith('UST'):
            continue
        s = p['hist'][a].iloc[-960:]
        gaps = s.index.to_series().diff().dt.days
        hole = max(10 * float(gaps.iloc[1:].median()), 5.0) if len(s) > 2 else 5.0
        st = s.diff()
        st[gaps > hole] = np.nan
        st = st.iloc[1:].dropna().to_numpy()
        if len(st) < 900:
            continue
        corr = REV_G0[fam] * (21.0 / h) ** REV_PW * h * (st[-600:-300].mean() + REV_C3 * st[-900:-600].mean())
        if np.isfinite(corr):
            d[j] -= float(np.clip(corr, -REV_CAP * sd[j], REV_CAP * sd[j]))
    return k * d


def transform_v8(p, fam, k, c, ct, ct2=None, beta=None, skew=None):
    base = p['base']
    sd = base.std(0, keepdims=True)
    sd = np.where(sd > 0, sd, 1.0)
    z = base / sd
    az = np.abs(z)
    ro = np.array([RISK_OFF.get(str(x).upper(), 0) for x, _ in p['cells']], float)[None, :]
    m = (1 + skew[0] * np.sign(z) * ro) if skew else 1.0
    ct2 = ct if ct2 is None else ct2
    f = cell_feats(p)
    ok = all(x is not None and x['v300'] > 0 for x in f)
    cm = 1.0
    if ok and fam in TREND_WIDTH:
        v3 = np.array([x['v300'] for x in f]); r60 = np.array([x['r60'] for x in f])
        lt = np.log(np.abs(r60) / (v3 * np.sqrt(60)) + 0.15)
        cm = np.exp(TREND_WIDTH[fam] * np.clip((lt - REGIME_NORM['lt60'][0]) / REGIME_NORM['lt60'][1], -2.5, 2.5))[None, :]
    zt = np.sign(z) * (c * cm * np.minimum(az, 1) + ct * m * np.clip(az - 1, 0, 1) + ct2 * m * np.maximum(az - 2, 0))
    zt = zt * vol_ratio(p, beta)[None, :]
    if skew:
        zt = zt + skew[1] * ro
    if ok and fam in REGIME_RULE:
        fe, sa, sb, eps, gam = REGIME_RULE[fam]
        v3 = np.array([x['v300'] for x in f])
        lv20 = np.log(np.maximum([x['v20'] for x in f], 1e-300) / v3)
        lv60 = np.log(np.maximum([x['v60'] for x in f], 1e-300) / v3)
        lv = {'lv20': lv20, 'lv60': lv60, 'lvm': 0.5 * (lv20 + lv60)}[fe]
        x = np.clip((lv - REGIME_NORM[fe][0]) / REGIME_NORM[fe][1], -2.5, 2.5)
        dr = np.asarray(p['drift'], float)
        shift = (sa + sb * x) * dr / (np.abs(dr) + eps * sd[0] + 1e-300)
        if gam:
            shift = shift * (np.array([h for _, h in p['cells']], float) / 63.0) ** gam
        zt = zt + shift[None, :]
    centre = p['anchor'] + reversal_drift(p, fam, k, sd[0])
    X = centre + sd * zt
    d = X.shape[1]
    m0 = p['anchor'] + p['drift'] + base
    if d == 2:
        if fam == 'F1' and p['cells'][0][0] == p['cells'][1][0]:
            D = X[:, 1] - X[:, 0]
            mp, mm = np.mean(np.abs(D) ** 0.5), np.mean(np.abs(m0[:, 1] - m0[:, 0]) ** 0.5)
            D = ((mm / mp) ** 2 if mp > 0 else 1.0) * D
            A = PAIR_LAM * X[:, 0] + (1 - PAIR_LAM) * X[:, 1]
            X = np.column_stack([A - (1 - PAIR_LAM) * D, A + PAIR_LAM * D])
        else:
            X = m0 + (X[:, :1] - m0[:, :1])
    elif fam == 'F3' and p['ttype'] == 'log_return' and d >= 3:
        ctr = centre[None, :]
        u = (X - ctr) / sd
        ub = u.mean(1, keepdims=True)
        dv = u - ub
        iu = np.triu_indices(d, 1)
        t0 = (np.abs(m0[:, iu[0]] - m0[:, iu[1]]) ** 0.5).mean(0)
        errs = []
        for t in TAU_GRID:
            Y = ctr + sd * (ub + t * dv)
            errs.append((((np.abs(Y[:, iu[0]] - Y[:, iu[1]]) ** 0.5).mean(0) - t0) ** 2).sum())
        X = ctr + sd * (ub + TAU_GRID[int(np.argmin(errs))] ** 0.5 * dv)
    return X


V12E = {'F1': (1.3, 1.0, 1, 1.3), 'F2': (1.6, 0.85, 1, 1.3), 'F3': (1.3, 1.0, 1, 1.3), 'F4': (1.6, 0.7, 0, 1.15),
        'em': (1.3, 1.2, 0, 1.0), 'macro': (1.15, 1.2, 0, 1.0)}
EM_K = 0.2
F4_ASYM = {'factors': (0.3, 0.1), 'rates': (0.15, 0.0)}
FACTORS = ('MKT', 'SMB', 'HML', 'MOM', 'QMJ', 'BAB')


def f4_asym(X, cells):
    out = X.copy()
    med = np.median(X, 0); sd = X.std(0); sd = np.where(sd > 0, sd, 1); u = (X - med) / sd
    for j, (a_, _) in enumerate(cells):
        name = str(a_).upper()
        cls = 'factors' if name in FACTORS else 'rates' if name.startswith('UST') else None
        if cls is None:
            continue
        ro = RISK_OFF.get(name, -1.0 if cls == 'rates' else 0.0)
        a, b = F4_ASYM[cls]
        out[:, j] = X[:, j] + sd[j] * (a * np.sign(u[:, j]) * ro * np.maximum(np.abs(u[:, j]) - 1, 0) + b * ro)
    return out


def reshape_v12e(X0, cells, m0, s, t, lock, g):
    X = X0 + (g - 1) * (X0.mean(0) - m0)
    med = np.median(X, 0); sd = X.std(0); sd = np.where(sd > 0, sd, 1)
    u = (X - med) / sd; au = np.abs(u)
    Y = med + sd * np.where(au <= 1, s * u, np.sign(u) * s * (1 + t * (au - 1)))
    if X.shape[1] == 2 and lock:
        i0, i1 = sorted(range(2), key=lambda i: (str(cells[i][0]), int(cells[i][1])))
        Y[:, i1] = Y[:, i0] + (X0[:, i1] - X0[:, i0])
    return Y


def params_for(card, panels_used):
    if any('macro' in n for n in panels_used):
        return dict(k=1.0, c=1.0, ct=1.0, kind='macro')
    if any('em_' in n or 'transfer' in n for n in panels_used):
        return dict(k=EM_K, c=1.0, ct=1.0, kind='em')
    m_ = re.search(r'F[1-4]', str(card.get('metadata', {}).get('category', '')))
    fam = m_.group(0) if m_ else str(card.get('metadata', {}).get('category', ''))[-2:]
    if fam in FAMILY_PARAMS:
        return dict(FAMILY_PARAMS[fam], fam=fam, skew=F4_SKEW if fam == 'F4' else None)
    return dict(PARAMS, vol=True)


def v9_card(p, fam, panels, asof):
    feats = []
    for f in cell_feats(p):
        feats.append(f if f is not None else dict(v20=0.0, v60=0.0, v300=0.0, r60=0.0))
    rates = []
    for _, df in panels:
        sub = df[df['asset'].str.upper().str.startswith('UST') & (df['date'] <= asof)]
        if len(sub):
            rates.append(sub)
    R = None
    if rates:
        R = pd.concat(rates).drop_duplicates(['date', 'asset'], keep='first').pivot(index='date', columns='asset', values='value').sort_index()
    cache = dict(p['hist'])

    def H(a):
        if a not in cache:
            cache[a] = history(panels, a, asof)
        return cache[a]
    return dict(fam=fam, ttype=p['ttype'], asof=asof, cells=[(str(a), int(h)) for a, h in p['cells']],
                anchor=np.asarray(p['anchor'], float), drift=np.asarray(p['drift'], float), base=p['base'],
                hs=np.array([h for _, h in p['cells']], float), feat=feats, H=cache, Hf=H, rates=R)


def ecb_calendar(hist, assets, asof):
    """True when the currency series carry rows on US federal holidays: the ECB fixes its reference rates on every TARGET day (US
    holidays included), the Fed's H.10 has no row on a US holiday.  Needs >= 5 such holidays in the trailing three years."""
    try:
        from pandas.tseries.holiday import USFederalHolidayCalendar
        dates = set()
        for a in assets:
            if str(a).upper() not in SHORT_FX or a not in hist:
                continue
            dates |= set(hist[a].index[hist[a].index >= asof - pd.Timedelta(days=1100)])
        if not dates:
            return False
        lo = min(dates)
        hol = [h for h in USFederalHolidayCalendar().holidays(start=lo, end=asof) if (h.month, h.day) not in ((1, 1), (12, 25))]
        return len(hol) >= 5 and sum(h in dates for h in hol) >= 0.6 * len(hol)
    except Exception:
        return False


SHORT_H = 20
SHORT_FX = ('AUD', 'CAD', 'CHF', 'DKK', 'EUR', 'GBP', 'JPY', 'NOK', 'NZD', 'SEK', 'MXN', 'BRL', 'ZAR', 'KRW', 'INR')
_UST_ALIAS = re.compile(r'(?:(UST|DGS|TSY|USGG|US|TREASURY|BILL|T)[-_. ]?)?(?:(?:T[-_. ]?)?BILLS?[-_. ]?|CMT[-_. ]?)?0*(\d+(?:\.\d+)?)[-_ ]?'
                        r'(MO|MON|MONTHS?|M|WKS?|W|WEEKS?|YRS?|Y|YEARS?)?(?:[-_. ]?CMT)?')
_FX_ALIAS = re.compile(r'(?:(?:FX|ECB)[-_.: ]?){0,2}(?:USD[-_/. ]?)?([A-Z]{3})(?:[-_/. ]?USD)?(?:[-_. ]?(?:ECB|RATE|SPOT|FIX|FIXING|CLOSE|PER[-_ ]USD|CURNCY))?(?:=?X)?')
_DEX = {'DEXUSEU': 'EUR', 'DEXUSUK': 'GBP', 'DEXUSAL': 'AUD', 'DEXUSNZ': 'NZD', 'DEXJPUS': 'JPY', 'DEXSZUS': 'CHF', 'DEXCAUS': 'CAD', 'DEXSDUS': 'SEK',
        'DEXNOUS': 'NOK', 'DEXDNUS': 'DKK', 'DEXMXUS': 'MXN', 'DEXBZUS': 'BRL', 'DEXSFUS': 'ZAR', 'DEXKOUS': 'KRW', 'DEXINUS': 'INR'}


def alias_id(x):
    """Model id for a series id spelled another way (UST_3MO, DGS3MO, TSY_1Y, USDMXN, MXN/USD, EUR_ECB ...): UST_<n>M / UST_<n>Y or the
    three-letter currency code; any other id (including a plain UST_2Y or EUR) comes back unchanged."""
    s = str(x).strip().upper()
    m = _UST_ALIAS.fullmatch(s)
    if m and (m.group(1) or m.group(3)):
        try:
            n, u = float(m.group(2)), (m.group(3) or 'Y')[0]
            months = n if u == 'M' else (n * 12 if u == 'Y' else round(n / 4.345))
            if m.group(1) == 'DGS' and not m.group(3):
                months = n * 12
            months = int(round(months))
            if months >= 12 and months % 12 == 0:
                return f'UST_{months // 12}Y'
            if months >= 1:
                return f'UST_{months}M'
        except Exception:
            pass
    m = _FX_ALIAS.fullmatch(s)
    if m and m.group(1) in SHORT_FX:
        return m.group(1)
    if s in _DEX:
        return _DEX[s]
    return x


def rename_panels(panels):
    """panels with every series id replaced by its alias_id (only when that cannot collide with an id already in the same panel)"""
    out = []
    for name, df in panels:
        ids = list(df['asset'].unique())
        ren = {a: alias_id(a) for a in ids}
        ren = {a: c for a, c in ren.items() if c != a}
        keep = set(ids) - set(ren)
        vals = list(ren.values())
        ok = {a: c for a, c in ren.items() if c not in keep and vals.count(c) == 1}
        out.append((name, df.assign(asset=df['asset'].map(lambda v: ok.get(v, v))) if ok else df))
    return out


def short_ust_fx(cells):
    return max(h for _, h in cells) <= SHORT_H and all(str(a).upper().startswith('UST') or str(a).upper() in SHORT_FX for a, _ in cells)


def declared_horizons(card, spec):
    """Short UST/FX level cards are in business days of a daily panel whose last row is the as-of: drop target dates so M0's walk takes h
    steps (M0-BASELINE 3.7 step 5 would count 1 step for h=2 on a Monday-Wednesday as-of, ratio exactly 2).  Returns (card, spec)."""
    try:
        t = card['targets']
        hs = t['horizons'] if isinstance(t['horizons'], (list, tuple)) else [t['horizons']]
        al = t['asset_ids'] if isinstance(t['asset_ids'], (list, tuple)) else [t['asset_ids']]
        cells = [(str(a), int(h)) for a in al for h in hs]
        if norm_ttype(t.get('target_type', 'level')) != 'level' or not cells or not short_ust_fx(cells):
            return card, spec
        drop = ('target_dates', 'observation_periods')
        card = {k: v for k, v in card.items() if k != 'questions'}
        card['targets'] = {k: v for k, v in t.items() if k not in drop}
        if isinstance(spec, dict):
            spec = {k: v for k, v in spec.items() if k != 'questions'}
            if isinstance(spec.get('targets'), dict):
                spec['targets'] = {k: v for k, v in spec['targets'].items() if k not in drop}
    except Exception:
        pass
    return card, spec


def short_market(p, prm):
    """short-horizon level card on daily series whose ids are not all recognised as Treasury/FX (and not a release or equity-factor card):
    it takes the v13w path with centres at the last level and no event timing for the unrecognised ids instead of the family reshape
    (swarm4/fresh route.py: on short UST/FX cards the v12e, em and default routes lose +0.06 to +0.46 per card, this one about 0)"""
    try:
        if p['ttype'] != 'level' or prm.get('kind') == 'macro' or max(h for _, h in p['cells']) > SHORT_H:
            return False
        for a in p['assets']:
            d = p['hist'][a].index[-WINDOW:]
            if str(a).upper() in FACTORS or len(d) < 20 or np.median(np.diff(d.values) / np.timedelta64(1, 'D')) > 3:
                return False
        return True
    except Exception:
        return False


def n_draws_min(card, spec):
    """minimum draw count declared by the card (card.toml [scoring.params] or forecast_spec.json), 0 when absent"""
    try:
        sp = spec.get('n_draws_min') if isinstance(spec, dict) else None
        return max(int(card.get('scoring', {}).get('params', {}).get('n_draws_min') or 0), int(sp or 0))
    except Exception:
        return 0


def forecast(card, spec, panels, asof, unit_id):
    p = m0_parts(card, spec, panels, asof, unit_id)
    used = set()
    for a in p['assets']:
        name, _ = asset_rows(panels, a)
        if name is not None:
            used.add(name)
    prm = params_for(card, used)
    ecb_fix = bool(card.get('_ecb_fx')) or any('ecb' in str(n).lower() for n in used) or ecb_calendar(p['hist'], p['assets'], asof)
    try:
        from . import v13r
        if v13r.applicable(p, card, spec, panels, asof):
            Xr, inf, st = v13r.draws(p, card, spec, panels, asof, unit_id)
            for c_, s_ in zip(p['cells'], st):
                p['steps'][c_] = s_
            p['drift'] = np.median(Xr, 0) - np.asarray(p['anchor'], float)
            return p, Xr, dict(prm, k=1.0, version='v13r', info_v13r=inf)
    except Exception as e:
        p['note_v13r'] = f'{type(e).__name__}: {str(e)[:80]}'
    nmin = n_draws_min(card, spec)
    known = short_ust_fx(p['cells']) and p['ttype'] == 'level'
    gen = not known and short_market(p, prm)
    if (known and not ('fam' in prm and prm['fam'] in V11.P)) or gen:
        try:
            from . import v13w
            cells = [(str(a), int(h)) for a, h in p['cells']]
            rec = {}
            Xw = v13w.draws(cells, p['anchor'], p['anchor'], p['base'], {a: p['hist'][a].to_numpy() for a in p['assets']}, p['Sig'],
                            [p['steps'][cell] for cell in p['cells']], p['assets'], pd.Timestamp(asof), ecb_fix=ecb_fix, unit=unit_id, n_min=nmin,
                            rec=rec)
            if np.isfinite(Xw).all():
                return p, Xw, dict(prm, version='v13wg' if gen else 'v13w0', **rec)
        except Exception as e:
            prm['note_v13w'] = f'{type(e).__name__}: {str(e)[:80]}'
    if 'fam' in prm and prm['fam'] in V11.P:
        prm = dict(prm, version='v11')
        try:
            c = v9_card(p, prm['fam'], panels, asof)
            X = V11.transform_card(dict(c, H=c['Hf'], unit=unit_id))
            if not np.isfinite(X).all():
                raise ValueError('non-finite draws')
            if short_ust_fx(c['cells']):
                vm = np.array([min(max((f['v60'] / f['v300']) ** 0.75, 0.5), 2.0) if f['v300'] > 0 and f['v60'] > 0 else 1.0 for f in c['feat']])
                v12c = X.mean(0)
                X13v = v12c[None, :] + np.asarray(c['base'], float) * vm[None, :]
                X, prm['version'] = X13v, 'v13v'
                try:
                    from . import v13w
                    cells = [(str(a), int(h)) for a, h in p['cells']]
                    if cells != list(c['cells']):
                        raise ValueError('cell order')
                    rec = {}
                    Xw = v13w.draws(cells, p['anchor'], v12c, c['base'],
                                    {a: p['hist'][a].to_numpy() for a in p['assets']}, p['Sig'], [p['steps'][cell] for cell in p['cells']],
                                    p['assets'], pd.Timestamp(asof), ecb_fix=ecb_fix, unit=unit_id, n_min=nmin, rec=rec)
                    if Xw.shape[1:] == X13v.shape[1:] and np.isfinite(Xw).all():
                        X, prm['version'] = Xw, 'v13w'
                        prm.update(rec)
                except Exception as e:
                    prm['note_v13w'] = f'{type(e).__name__}: {str(e)[:80]}'
            else:
                X = reshape_v12e(X, c['cells'], c['anchor'] + c['drift'], *V12E[prm['fam']])
                if prm['fam'] == 'F4':
                    X = f4_asym(X, c['cells'])
                prm['version'] = 'v12e'
        except Exception as e:
            prm['degraded'] = f'Primary method failed (v11 transform: {type(e).__name__}: {str(e)[:120]}); '
            try:
                X = transform_v8(p, **{k: v for k, v in prm.items() if k not in ('version', 'degraded')})
                if not np.isfinite(X).all():
                    raise ValueError('non-finite draws')
                prm['degraded'] += 'v8 transform used instead.'
            except Exception:
                X = p['anchor'] + p['drift'] + p['base']
                prm['degraded'] += 'exact M0 draws used instead.'
    elif 'kind' in prm:
        X = transform(p, **{k: v for k, v in prm.items() if k != 'kind'})
        X = reshape_v12e(X, p['cells'], np.asarray(p['anchor'], float) + np.asarray(p['drift'], float), *V12E[prm['kind']])
        prm['version'] = 'v12e'
    else:
        X = transform_v8(p, **prm) if 'fam' in prm else transform(p, **prm)
    if not np.isfinite(X).all():
        raise ValueError('non-finite draws')
    return p, X, prm


def m0_draws(card, spec, panels, asof, unit_id):
    p = m0_parts(card, spec, panels, asof, unit_id)
    X = np.asarray(p['anchor'], float) + np.asarray(p['drift'], float) + p['base']
    if X.shape != (N_DRAWS, len(p['cells'])) or not np.isfinite(X).all():
        raise ValueError('non-finite M0 draws')
    return p['cells'], X


def _has_history(panels, a, asof):
    try:
        return len(history(panels, a, asof)) >= 3
    except Exception:
        return False


def safe_fallback(card, spec, panels, asof, unit_id):
    """Tiered last resort, never raises on a readable grid: (1) the exact M0 baseline draws; (2) M0 on the assets that have history
    and the independent walk on the others; (3) the independent walk.  Returns (cells, X, tier description)."""
    t = card['targets']
    try:
        cells, X = m0_draws(card, spec, panels, asof, unit_id)
        return cells, X, 'exact M0 baseline draws'
    except Exception:
        pass
    try:
        assets = sorted({str(a) for a in t['asset_ids']})
        ok = [a for a in assets if _has_history(panels, a, asof)]
        if 0 < len(ok) < len(assets):
            cells_ok, X_ok = m0_draws(dict(card, targets=dict(t, asset_ids=ok)), spec, panels, asof, unit_id)
            cells, X = fallback(card, panels, asof)
            col = {c: j for j, c in enumerate(cells_ok)}
            for j, c in enumerate(cells):
                if c in col:
                    X[:, j] = X_ok[:, col[c]]
            return cells, X, 'M0 baseline draws on the assets with history and the independent Gaussian walk on the rest'
    except Exception:
        pass
    cells, X = fallback(card, panels, asof)
    return cells, X, 'independent Gaussian walk fallback'


def finite_or_fallback(X, cells, card, panels, asof):
    """belt and braces: any non-finite column (or a shape mismatch) is replaced by the independent walk; returns (cells, X)"""
    if isinstance(X, np.ndarray) and X.ndim == 2 and X.shape[1] == len(cells) and X.shape[0] >= MIN_DRAWS and np.isfinite(X).all():
        return cells, X
    cells_fb, Xf = fallback(card, panels, asof)
    if not (isinstance(X, np.ndarray) and X.shape == (N_DRAWS, len(cells)) and list(cells) == list(cells_fb)):
        return cells_fb, Xf
    X = X.copy()
    bad = ~np.isfinite(X).all(axis=0)
    X[:, bad] = Xf[:, bad]
    return cells, X


def fallback(card, panels, asof):
    """Last resort: independent Gaussian walk from each asset's last value."""
    t = card['targets']
    ttype = norm_ttype(t.get('target_type', 'level'))
    cells = sorted((str(a), int(h)) for a in t['asset_ids'] for h in t['horizons'])
    rng = np.random.default_rng(0)
    X = np.empty((N_DRAWS, len(cells)))
    for j, (a, h) in enumerate(cells):
        try:
            s = history(panels, a, asof).dropna()
            if ttype != 'level' and len(s) > 2 and float(np.median(np.abs(s.to_numpy()))) >= 0.2:
                v = (np.log(s) if (s > 0).all() else s).diff().dropna()
            else:
                v = s.diff().dropna() if ttype == 'level' else s
            sd = float(v.iloc[-WINDOW:].std()) if len(v) > 2 else 0.0
            sd = sd if np.isfinite(sd) and sd > 0 else max(abs(float(s.iloc[-1])) * 0.01, 1e-4)
            last = float(s.iloc[-1]) if ttype == 'level' else 0.0
        except Exception:
            sd, last = 1.0, 0.0
        X[:, j] = last + sd * np.sqrt(h) * rng.standard_normal(N_DRAWS)
    return cells, X
