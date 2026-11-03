import argparse, json, pathlib, signal, sys, tomllib, traceback
import numpy as np
import pandas as pd
from . import core


def rationale(unit_id, asof, p, prm, n_docs, note, n_draws=core.N_DRAWS):
    grid = ('each cell\'s draws are a variance-preserving Student-t(12) quantile grid with that sd, placed by the ranks of the baseline\'s '
            'joint draws under a t(2) copula' if not prm.get('d1') else
            'each cell\'s draws are a variance-preserving Student-t(12) quantile grid with that sd at the positions that are optimal for the '
            'fair CRPS estimator, placed by the ranks of a deterministic low-discrepancy (Halton) sample of the model correlation under a t(2) copula')
    lines = [f'# Forecast rationale — {unit_id}', '',
             f'As of **{asof}**. {n_draws} joint draws.', '',
             '## Method', '',
             '1. Statistical backbone: joint Gaussian random walk on the trailing 300 panel steps '
             '(level changes, or per-step returns for log-return targets), steps spanning data holes dropped, '
             'assets aligned by date, cross-horizon path covariance min(s, s\') * Sigma.',
             f'2. Drift: the trailing-window mean step is shrunk by k = {prm["k"]}; for US Treasury yields the drift is '
             'reduced by part of the trend from 300-900 steps earlier (yields partly reverse multi-year trends).',
             f'3. Shape: standardized draws are reshaped piecewise-linearly, slope {prm["c"]} inside one sd and '
             f'{prm["ct"]} beyond it (h-day financial changes are leptokurtic); width scaled by recent vs 300-step volatility.',
             '   The centre leans slightly along the trailing drift, more when recent volatility is elevated.',
             '   Tail/shock (F4) cards: equity-factor and currency targets get a heavier risk-off tail and a 0.25 sd risk-off lean.'
             if prm.get('skew') else '   No directional risk-off skew on this card family.',
             '4. Parameters were chosen on pre-as-of historical backtests only; no unit-specific values are stored.',
             (f'5. Calibration for errors measured against the baseline\'s expected error: the shift of the centre away from the baseline mean '
              f'is scaled by {core.V12E[prm.get("fam", prm.get("kind"))][3]}, deviations from each cell median by {core.V12E[prm.get("fam", prm.get("kind"))][0]} '
              f'within one sd and the tail slope by a further {core.V12E[prm.get("fam", prm.get("kind"))][1]}.'
              + (' Tail/shock card: equity-factor cells get a heavier risk-off tail (+0.3) and a 0.1 sd risk-off lean; '
                 'Treasury-yield cells a heavier falling-yield tail (+0.15).' if prm.get('fam') == 'F4' else ''))
             if prm.get('version') == 'v12e' else
             ('5. Short-horizon Treasury/G10 FX card: the centre above is kept and the baseline width is scaled by '
              'clip((60-step / 300-step volatility)^0.75, 0.5, 2) per cell.' if prm.get('version') == 'v13v' else
              ('5. Short-horizon Treasury/FX card: Treasury bill centres keep the centre above, note centres move half of the way and currency '
               'centres only a quarter of the way from the last level to it; per-asset width = baseline width x clip((EWMA(0.94) sd / 300-step sd)^0.8, '
               '0.5, 2), 0.35-3 for bills; scheduled-event timing (FOMC, US payrolls, BoJ, ECB calendars; separate Treasury bill, note, G10 and EM '
               f'tables) rescales each horizon increment by its expected variance relative to the trailing 300 business days; {grid}.'
               if prm.get('version') == 'v13w' else
              ('5. Short-horizon Treasury/FX card outside the family path: centres at the last level; per-asset width = baseline width x '
               'clip((EWMA(0.94) sd / 300-step sd)^0.8, 0.5, 2); scheduled-event timing (FOMC, US payrolls, BoJ, ECB) rescales each horizon '
               f'increment; {grid}.' if prm.get('version') == 'v13w0' else
              ('5. Short-horizon card on daily series whose ids are not all recognised: centres at the last level; per-asset width = baseline '
               'width x clip((EWMA(0.94) sd / 300-step sd)^0.8, 0.5, 2); scheduled-event timing only for recognised Treasury/FX series; '
               f'{grid}.' if prm.get('version') == 'v13wg' else
              ('5. Official-data release card (weekly / monthly / quarterly series, horizon <= 20 BD): the number of series periods to the target '
               'period is counted from the last observation; centre = inverse-error combination of window-mean predictors on the recent '
               'self-forecast errors, with a same-period-last-year predictor when the series is unadjusted (PCE: regression on the matching CPI '
               'series for the missing month(s) when the panel holds it); width = robust sd of those errors, '
               'Student-t (4 dof) shape on an exact quantile grid.' if prm.get('version') == 'v13r' else '5. No extra calibration on this card.'))))),
             '', '## Ledger', '', '| asset | horizon | anchor | drift used | panel steps |', '|---|---|---|---|---|']
    if p is not None and prm.get('version') in ('v13w', 'v13w0', 'v13wg'):
        from . import v13w
        pr = v13w.FINAL_FOMC_PRIOR
        if pr is not None:
            gap = v13w.BD.searchsorted(pr['decision']) - v13w.BD.searchsorted(pd.Timestamp(asof))
            fp = {}
            for a, h in p['cells']:
                k = v13w._final_prior(a, asof)
                if k is not None and 1 <= gap <= int(h):
                    fp[str(a)] = k
            if fp:
                j = next(i for i, x in enumerate(lines) if x.startswith('5. '))
                lines[j] += (f' Disclosed judgment prior for the {pr["decision"]:%d %b %Y} FOMC statement day: the sd of that day\'s increment in the '
                             'Treasury bill cells is multiplied by ' + ', '.join(f'{v:g} ({a})' for a, v in fp.items()) + ', set from market-implied '
                             'odds of a move at that meeting read before the as-of (the event tables were fitted on mostly priced meetings); it is '
                             'a forward-looking judgment, not fitted to realized values.')
    if p is not None:
        for j, (a, h) in enumerate(p['cells']):
            lines.append(f'| {a} | {h} | {p["anchor"][j]:.6g} | {prm["k"] * p["drift"][j]:.6g} | {p["steps"][(a, h)]} |')
    lines += ['', '## Text', '', f'{n_docs} document(s) available; this version does not adjust on text.']
    if note:
        lines += ['', '## Notes', '', note]
    return '\n'.join(lines) + '\n'


def _write(a, unit_id, asof_s, t, cells, X, p, prm, note):
    a.out.parent.mkdir(parents=True, exist_ok=True)
    n, d = X.shape
    df = pd.DataFrame({
        'draw': np.repeat(np.arange(n, dtype=np.int32), d),
        'asset': pd.array([c[0] for c in cells] * n, dtype='string'),
        'horizon': np.tile(np.array([c[1] for c in cells], dtype=np.int32), n),
        'value': X.reshape(-1).astype(np.float64),
    })
    df.to_parquet(a.out, index=False)
    meta = {'unit_id': unit_id, 'asof': asof_s, 'representation': 'samples',
            'asset_ids': [str(x) for x in t['asset_ids']], 'horizons': [int(h) for h in t['horizons']],
            'n_draws': int(n)}
    if t.get('target_type', 'level') in META_TARGETS:
        meta['target'] = t.get('target_type', 'level')
    meta['rationale'] = {'file': 'forecast_rationale.md', 'method': 'M0-backbone RW, shrunk drift, leptokurtic reshape'}
    if t.get('value_unit') and isinstance(t['value_unit'], str):
        meta['units'] = t['value_unit']
    (a.out.parent / 'forecast_meta.json').write_text(json.dumps(meta, indent=2) + '\n')
    n_docs = len(list(a.text.glob('*'))) if a.text and a.text.is_dir() else 0
    try:
        rat = rationale(unit_id, asof_s, p, prm, n_docs, note, n)
    except Exception:
        rat = f'# Forecast rationale - {unit_id}\n\nAs of {asof_s}. Draws from the M0-backbone random walk (fallback text).\n\n' + (note or 'No note.') + '\n'
    (a.out.parent / 'forecast_rationale.md').write_text(rat)
    print(f'wrote {a.out} ({n} draws x {d} cells){" [fallback]" if note else ""}')
    return 0


def _ecb_text(x):
    """a panel source string that names the ECB euro foreign exchange reference rates (any plausible wording)"""
    x = str(x).lower()
    named = 'ecb' in x or 'european central bank' in x or 'eurofxref' in x or 'eurosystem' in x
    return (named and ('reference' in x or 'exchange' in x or 'eurofxref' in x or 'exr' in x or 'fixing' in x)) or ('14:10' in x and ('cet' in x or 'euro' in x))


def _panel_sources(card, spec):
    """source strings of the panels as declared in card.toml [panels.*] and forecast_spec.json panels[] (the manifest is read first)"""
    out = []
    pn = card.get('panels')
    if isinstance(pn, dict):
        out += [str(v.get('source', '')).lower() for v in pn.values() if isinstance(v, dict)]
    if isinstance(spec, dict) and isinstance(spec.get('panels'), list):
        out += [str(v.get('source', '')).lower() for v in spec['panels'] if isinstance(v, dict)]
    return out


META_TARGETS = ('level', 'log_return', 'yield')
MODEL_BUDGET_S = 1000


class _Budget(BaseException):
    pass


def _on_alarm(signum, frame):
    raise _Budget('modelling time budget exceeded')


def _arm(seconds):
    """time budget for the modelling step (the unit limit is 1,800 s); no-op where SIGALRM does not exist"""
    try:
        if seconds:
            signal.signal(signal.SIGALRM, _on_alarm)
        signal.setitimer(signal.ITIMER_REAL, seconds)
    except Exception:
        pass


def _parse_asof(arg, card):
    try:
        return pd.Timestamp(str(arg)[:10])
    except Exception:
        return pd.Timestamp(str(card['provenance']['data_cutoff'])[:10])


def main(argv=None):
    ap = argparse.ArgumentParser(prog='forecast')
    ap.add_argument('verb', nargs='?')
    ap.add_argument('--panels', type=pathlib.Path, required=True)
    ap.add_argument('--text', type=pathlib.Path, default=None)
    ap.add_argument('--asof', required=True)
    ap.add_argument('--out', type=pathlib.Path, required=True)
    ap.add_argument('--card', type=pathlib.Path, default=None)
    a = ap.parse_known_args(argv)[0]
    try:
        return _run(a)
    except Exception:
        tb = traceback.format_exc()
        sys.stderr.write(tb)
        return _emergency(a, tb)


def _run(a):
    try:
        card_path = a.card or core.find_card(a.panels)
        card = tomllib.loads(card_path.read_text(encoding='utf-8-sig'))
        card_dir = card_path.parent
        card['targets']['asset_ids'], card['targets']['horizons']
    except Exception:
        card_dir = next((d for d in (a.panels, a.panels.parent) if (d / 'forecast_spec.json').exists()), None)
        if card_dir is None:
            raise
        s_ = json.loads((card_dir / 'forecast_spec.json').read_text())
        card = {'task': {'id': s_.get('card_id')}, 'metadata': {'category': s_.get('card_family', '')}, 'targets': s_['targets']}
    spec = None
    sp = card_dir / 'forecast_spec.json'
    if sp.exists():
        try:
            spec = json.loads(sp.read_text())
        except Exception:
            spec = None
    try:
        man = json.loads((card_dir / 'manifest.json').read_text())
        srcs = [str(f.get('source', '')).lower() for f in man.get('files', []) if str(f.get('path', '')).endswith('.parquet')]
        card['_ecb_fx'] = any(_ecb_text(x) for x in srcs)
    except Exception:
        pass
    if not card.get('_ecb_fx'):
        try:
            card['_ecb_fx'] = any(_ecb_text(x) for x in _panel_sources(card, spec))
        except Exception:
            pass
    unit_id = str(card.get('task', {}).get('id') or card_dir.name)
    asof = _parse_asof(a.asof, card)
    asof_s = asof.strftime('%Y-%m-%d')
    panels = core.read_panels(a.panels)
    if isinstance(spec, dict) and isinstance(spec.get('targets'), dict) and isinstance(spec['targets'].get('as_of_level'), dict):
        panels = core.add_asof_levels(panels, spec['targets']['as_of_level'], asof)
    panels = core.rename_panels(panels)
    t = card['targets']
    if isinstance(t.get('asset_ids'), str):
        t['asset_ids'] = [t['asset_ids']]
    if not isinstance(t.get('horizons'), (list, tuple)):
        t['horizons'] = [t['horizons']]
    canon = {}
    for a_ in t['asset_ids']:
        _, rows_ = core.asset_rows(panels, str(a_))
        canon[str(a_)] = str(rows_['asset'].iloc[0]) if len(rows_) else str(a_)
    if len(set(canon.values())) != len(canon):
        canon = {k: k for k in canon}
    back = {v: k for k, v in canon.items()}
    card_run = card
    if any(k != v for k, v in canon.items()):
        card_run = dict(card, targets=dict(t, asset_ids=[canon[str(x)] for x in t['asset_ids']]))
    card_run, spec = core.declared_horizons(card_run, spec)
    note, p, prm = '', None, dict(k=1.0, c=1.0, ct=1.0)
    try:
        _arm(MODEL_BUDGET_S)
        p, X, prm = core.forecast(card_run, spec, panels, asof, unit_id)
        cells = p['cells']
        note = prm.get('degraded', '')
    except (Exception, _Budget):
        tb = traceback.format_exc()[-1500:]
        _arm(0)
        cells, X, tier = core.safe_fallback(card_run, spec, panels, asof, unit_id)
        note = f'Primary method failed; {tier} used.\n\n```\n' + tb + '\n```'
        p, prm = None, dict(k=1.0, c=1.0, ct=1.0)
    finally:
        _arm(0)
    cells, X = core.finite_or_fallback(X, cells, card_run, panels, asof)
    cells = [(back.get(c[0], c[0]), c[1]) for c in cells]
    return _write(a, unit_id, asof_s, t, cells, X, p, prm, note)


def _emergency(a, tb):
    """Anything outside the guarded modelling block raised: rebuild the grid from whatever is readable and write a complete, valid
    output (M0-equivalent draws, else the independent walk), so the unit is scored rather than counted as a crash."""
    try:
        card = card_dir = spec = None
        for path in ([a.card] if a.card else []) + [a.panels / 'card.toml', a.panels.parent / 'card.toml']:
            try:
                c_ = tomllib.loads(path.read_text(encoding='utf-8-sig'))
                c_['targets']['asset_ids'], c_['targets']['horizons']
                card, card_dir = c_, path.parent
                break
            except Exception:
                continue
        for d in ([card_dir] if card_dir else []) + [a.panels, a.panels.parent]:
            try:
                spec = json.loads((d / 'forecast_spec.json').read_text())
                card_dir = card_dir or d
                break
            except Exception:
                continue
        if card is None:
            s_ = spec if isinstance(spec, dict) else {}
            if not isinstance(s_.get('targets'), dict):
                raise RuntimeError('neither card.toml nor forecast_spec.json gives the target grid')
            card = {'task': {'id': s_.get('card_id')}, 'metadata': {'category': s_.get('card_family', '')}, 'targets': s_['targets']}
        t = card['targets']
        if isinstance(t.get('asset_ids'), str):
            t['asset_ids'] = [t['asset_ids']]
        if not isinstance(t.get('horizons'), (list, tuple)):
            t['horizons'] = [t['horizons']]
        unit_id = str(card.get('task', {}).get('id') or (card_dir.name if card_dir else 'unit'))
        try:
            panels = core.read_panels(a.panels)
        except Exception:
            panels = []
        asof = _parse_asof(a.asof, card)
        card, spec = core.declared_horizons(card, spec)
        cells, X, tier = core.safe_fallback(card, spec, panels, asof, unit_id)
        cells, X = core.finite_or_fallback(X, cells, card, panels, asof)
        note = f'Primary method failed; {tier} used.\n\n```\n' + tb[-1500:] + '\n```'
        return _write(a, unit_id, asof.strftime('%Y-%m-%d'), t, cells, X, None, dict(k=1.0, c=1.0, ct=1.0), note)
    except Exception:
        sys.stderr.write(traceback.format_exc())
        return 1


if __name__ == '__main__':
    sys.exit(main())
