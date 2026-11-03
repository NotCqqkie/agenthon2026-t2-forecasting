"""v11 transform, ported from t2-work/candidates/v11_final.py (docstring below lists components and constants).
History: c['H'](asset) -> Series up to as-of from the unit's own panels; c['rates'] -> the unit's rates-panel levels up to as-of."""
"""v11_final = v10_final + round-4 components accepted by greedy per-family forward selection (dev sets, evalkit; accept rule: dev W -0.0015 or better, W_early / W_late not
worse, real not worse by > 0.003, >= 4 of 6 four-year blocks improve, no block worse by > 0.01, final-like (as-of UST_2Y in [2.5, 5.5]) W not worse).

Card fields used (all as of c['asof']): fam, ttype, asof, unit, anchor, drift, base, hs, feat (v20/v60/v300/r60) and the unit's own panel histories
(harness.SERIES through the helpers below, sliced to dates <= asof).  Never c['y'] / c['ref'].  Missing history degrades to v10 behaviour (component adds 0).

V11 COMPONENTS (per family; constants in COMPONENTS / SELECTED / P0 / CFG0 below; shift is in units of the cell's M0 sd: X = centre + sd*(... + shift))
 F1  joint     step 9, d == 2 cards (one-asset 126/189 pair): exact-variogram rank coupling that keeps v10's marginals (see JOINT below).  THETA .5, LAM_F1 .3
     (everything else = v10_final)
 F2  wild      step 7c: front-end momentum f = f2m_60 (below); x = clip((f - .27)/.88, +-2.5); UST level cells: shift += PD*x*(h/63)**PDH*max(1 + PDS*log(10/tenor)/log5, 0)
               PD .15  PDH .5  PDS .3
     f2r_rev   PW .5 -> .35 (UST reversal strength exponent in (21/h)**PW), rtail .5 -> .75
     f2r_fx    fx_cap .75 (step 7b-W now also for F2: G10-FX level cells, centre drift clipped at +-.75 sd), plus step 7b-M "fxm" (5, .1, 2.5): G10-FX level cells
               shift += .1*clip(S5/(s300*sqrt 5), +-2.5)*sqrt(h/63), S5 = sum of the last 5 steps of the cell's own asset, s300 = sd of its last 300 steps (steps_of)
     f2r_dt    DP tail coefficient DT .09 -> .15  (DCEN .75, DW .02 unchanged)
 F3  wild      as F2 with PD .10 PDH .5 PDS .3, plus PDX .15: in cards with a UST level cell AND G10-FX cells, FX cells shift += PDX*s_a*x*(h/63)**PDH
               (x = clip((f2m_60 - .27)/.88, +-2.5); s_a = +1 for foreign-per-USD quotes CAD CHF DKK JPY NOK SEK, -1 for USD-per-foreign AUD EUR GBP NZD)
     vol4      width factor vr = clip(exp(.25*e1 + .15*e2), vlo .7, vhi 1.4) replacing (v20/v300)**beta: per cell e1 = log(ewma rms / sd300), e2 = log(sd1500 / sd300) of the
               cell's own last <= 3300 clean steps, ewma lam = clip(1 - 4/h, .85, .997); 0 where unavailable (e2 needs >= 1500 steps)   (kap 4, b .25, lt .15)
 F4  f4shock   five additions (hp = sqrt(h/21)):
     f4s_w10     UST_10Y level cells: width *= exp(W10), W10 -.12
     f4s_r10     UST_10Y level cells: shift += R10*r250*hp, r250 = clip(sum of last 250 own steps/(sd300*sqrt 250), +-3) (r250_of), R10 -.15
     f4s_z2      UST_2Y level cells: shift += Z2*zs*hp, zs = smoothstep((UST_2Y level - 1)/2) from the rates panel (0 if unavailable) (zs_2y), Z2 -.4
     f4s_lv5     equity-factor log_return cells: shift += LV5*clip(log(rms(last 5 own steps)/sd300)/.55, +-3) (lv5_of), LV5 -.2
     f4s_ros     ro (+-1 risk-off sign used by a=.3 tail asymmetry and b=.25 centre shift) multiplied per asset: JPY .5, MKT 1.5
     joint     step 9, d == 2 cards (MKT/QMJ pair): as F1; the infeasible-target fallback is fix_scale with LAM_OTHER .5
 Not selected (tested, rejected): F4 wild4 (UST momentum + y2 + mean_r500 terms; after f4shock dW -.0025 but W_early +.0008 and real +.0037; reduced / re-weighted variants
 and Z2 re-tuning also failed or passed only marginally), joint on F2 (dW -.0008), joint on F3 (no d == 2 cards: no change).

JOINT (step 9, d == 2): both cells keep v10's marginals (sorted values); only the draw-to-draw pairing of cell 1 against cell 0 changes: Gaussian-copula pairing in the
normal scores of M0's draws (c['base']/sd), correlation rho by bisection (40 steps) so mean|X1-X0|^.5 equals M0's (mm); then the increment is rescaled by (mm/mp)**2 (~1) so the
variogram term stays ratio 1.  Infeasible targets: F1 fix_theta (increment re-centred toward M0's mean difference with weight THETA, rescaled to hit mm, cells re-centred
with weight LAM_F1; falls back to fix_scale); F4 fix_scale (rescale, re-centre with LAM_OTHER).

PANEL FEATURE f2m_60 (panel_feats): r_T(60) = (level_T[asof] - level_T[asof-60 rows]) / (sd300_T*sqrt 60), sd300 = 300-row rolling std (min 200) of the daily changes of the
ffilled level; f2m_60 = r_UST_2Y - mean of r_T over UST_5Y 7Y 10Y 20Y 30Y.  Needs the rates-panel row within 8 days of asof, >= 600 rows, all six tenors with sd300 > 0, else {}.

V10 PIPELINE (still the base of v11; [v10] = added in v10)
 0  zlb gate [v10]  w = (1-S((y2-ZA0)/(ZA1-ZA0)))*(1-S((ZS0-slope)/(ZS0-ZS1))), S = smoothstep on [0,1], y2 = UST_2Y level, slope = UST_10Y-UST_2Y at asof
                    (rates_daily panel, last obs within 8 days of asof; w = 0 if unavailable).  Applied to UST level cells only: G0 *= (1-w) in step 2
                    (so rtail / a_cabs / wv_ldzc, which read the reversal, shrink with it) and ltail *= (1-w) in step 4.  ZA0 .75 ZA1 1.75 ZS0 2.2 ZS1 1.2  (all families)
 1  v7 reshape      zt = sign(z)*(c*min(|z|,1) + ct*m*clip(|z|-1,0,1) + ct2*m*max(|z|-2,0)) [tail parts also * (1+DT*xd), step D];  X = centre + sd*(zt*vr*vm + b*ro + shift)
                    centre = anchor + k*KMvec*drift_adj + ksl term (step K);  KMvec = KM on equity-factor cells (MKT SMB HML MOM QMJ BAB, log_return), 1 elsewhere
                    vr = clip((v20/v300)**beta, vlo, vhi); m = 1 + a*sign(z)*ro (ro = +-1 risk-on/off sign of the asset; 0 for UST); core c scaled by
                    exp(slope*clip((log(|r60|/(v300*sqrt60)+.15)+.22)/.68, +-2.5))
 2  UST reversal    UST level cells: corr = G0*(1-w)*(21/h)**PW*gten*h*mj, |corr|<=CAP*sd; mj = m[300,600)+.5*m[600,900) (m = mean step change that many steps back);
                    drift_adj = drift - corr  (other cells: drift_adj = drift; needs >= 900 steps).  gten = max(1+gslope*log(10/tenor)/log5, 0)
                    [v10, step L, F3 only] slope-factor scaling: when the card has >= 2 UST assets, mj of tenor T is decomposed over the six tenors (UST_2Y..30Y,
                    those with >= 900 steps, need >= 3): mj_T = a0 + b0*x_T + res_T with x_T = log T - mean(log T) (OLS); used mj_T = a0 + (h/63)**lamS*b0*x_T + res_T
 3  rtail           UST level cells: tail multiplier m *= 1 + rtail*sign(z)*clip(-corr/sd, +-1)   (tail heavier on the side the reversal moved the centre)
 4  ltail           UST level cells: m *= 1 + ltail*(1-w)*sign(z)*clip((MA1500 - anchor)/3, +-1)
 5  regime lean     shift = (sa + sb*clip((lv-med)/scale, +-2.5)) * dr/(|dr| + eps*sd) * (h/63)**gam,   lv = log(v20/v300), log(v60/v300) or their mean,
                    dr = drift_adj when shiftadj else M0 drift
 6  regime terms    x_f = clip((f-centre)/scale, +-2.5):
                    wv_ldzc : width *= exp(val*x), f = log(|drift_adj/sd|+.15)                          (F1)
                    a_cabs  : shift += val*x*dirdr, f = |corr|/sd, dirdr = drift/(|drift|+.1*sd)        (F1)
                    a_lvlN  : shift += val*x*dirdr, f = lvl on non-UST cells (0 on UST level cells), lvl = log(sd1500/sd300) of the cell's own steps (F2)
                    wv_lvlN : width *= exp(val*x), same f                                               (F2)
                    m_lvl   : shift += val*x*clip(drift_adj/sd, +-3), f = lvl                           (F4)
 7  xasset (UST)    UST cells: shift += (h/63)**.5 * sum coef*clip(feat, +-3);  u10_n = (sum of last n UST_10Y daily changes)/(sd300(UST_10Y)*sqrt n)
                    lvpm = pooled over all rates assets of 0.5*(log v20/v300 + log v60/v300), times lean = dsd/(|dsd|+.1), dsd = drift/sd
 7b [v10] additions (applied to shift / width / centre after step 7):
    W  fx_cap (F1)  G10-FX level cells (AUD NZD EUR GBP CAD NOK SEK DKK JPY CHF): shift += k*(clip(drift_adj, +-FXCAP*sd) - drift_adj)/sd   (FXCAP 1.0)
    V  CV (F1, F4)  equity-factor cells (MKT SMB HML MOM QMJ BAB, log_return): shift -= CV*clip(0.5*(log v20/v300 + log v60/v300), +-3)*sqrt(h/63)
    D  DP (F2, F3)  xd = clip((|drift_adj/sd|/sqrt(h/63) - DCEN)/.5, +-2.5): width *= exp(DW*xd); tail parts of zt * (1 + DT*xd)       (DCEN .75 DW .02 DT .09)
    K  ksl (F2, F3) UST level cells: centre += (f-1)*k*drift (raw M0 drift), f = max(1 + ksl*log(5/tenor)/log(2.5), 0)  (steeper extrapolation of the drift at short tenors)
 8  yield floor     UST level cells: delta<0 -> -L*tanh(-delta/L), L = 2*max(anchor-0.2, 0.05)
 9  pair/variogram  d==2: M0's pairwise differences kept (one-asset F1 pair: increment rescaled so mean|D|^.5 equals M0's, re-centred with weight LAM);
                    F3 log_return d>=3: deviations from the cross-sectional mean scaled by tau**TEXP (tau matches M0's pair dispersion)

V10 BASE CONSTANTS per family (P0 / CFG0 below; v11 changes are listed above)
 F1  k .625 c .5 ct 1.0 ct2 1.15 beta .25 a .1 b -.05 slope .05 vlo .7 vhi 1.4 | rule lvm sa .069 sb .0645 eps .1 gam 0 | G0 .75 PW .5 CAP 2.5
     gslope 0 rtail 0 ltail .3 floor | LAM .3 | wv_ldzc .1 a_cabs .0625 | xasset UST u10_750 coef -.07 | [v10] zlb; V CV .2; W fx_cap 1.0
 F2  k .5 c 1.0 ct 1.3 ct2 1.3 beta .25 | rule lvm sa -.026 sb .178 eps .1 gam 0 | G0 1.0 PW .5 CAP 2.5 gslope 0 rtail .5 floor
     a_lvlN .3 wv_lvlN .125 | xasset UST u10_500 coef -.12 | [v10] zlb; D (.75,.02,.09); K ksl .2
 F3  k .375 c .9 ct 1.3 ct2 1.3 beta .25 | rule lvm sa .094 sb 0 eps .1 gam 0 | G0 1.0 PW .5 CAP 2.5 gslope .3 rtail .5 floor | TEXP .5
     xasset UST u10_750 coef -.135, lvpm coef .36 | [v10] zlb; D (.75,.02,.09); K ksl .3; L lamS .5
 F4  k .625 c 1.6 ct 1.5[v10, was 1.7] ct2 1.7 beta .4375[v10, was .5] a .3 b .25 slope .106 | rule lv60 sa .0608 sb .3735 eps .1 gam .5, shiftadj
     G0 1.4 PW 0 CAP 4.0 gslope .3 rtail .5 floor | m_lvl .1 | no xasset | [v10] zlb; V CV .2 KM .7; robust ct/beta above

HELPERS: all history access goes through hist() (levels of an asset from the unit's own panel that holds it, dates <= asof) or the rates-panel functions.
 hist(c, a, n)       asset a's last n levels <= asof from asset_file(a, unit panels)    [the only reader of per-asset series]
 steps_of(c, a)      hist(c, a, 3360) -> clean diffs (gaps > 10x median spacing dropped), last <= 3300.  Called for: each UST cell's asset (reversal, UST level cards, all
                     families); F3: all six tenors UST_2Y 5Y 7Y 10Y 20Y 30Y (lamS; those missing from the unit's panels are skipped); F2 fxm: each G10-FX cell's own asset
 ma1500(c)           hist(c, a, 1500) of every cell's own asset: mean (nan if < 500 obs)                                    [F1 ltail, only UST cells matter]
 lvl_of(c, j)        hist(c, a, 1700) of cell j's own asset -> log(sd last 1500 steps / sd last 300)                         [F2 a_lvlN / wv_lvlN, F4 m_lvl]
 own_steps(c, a)     hist(c, a, 1700) of cell asset a: np.diff (level cards) or the values (log_return), last <= 1500; r250_of: UST_10Y cells; lv5_of: equity-factor cells
                     [F4 only: f4s_r10 / f4s_lv5]
 ew_log_ratio(c, j, kap)  hist(c, a, 3360) of cell j's own asset                                                            [F3 vol4]
 _rates_panel()      all assets of the rates_daily panel (the unit's rates panel, present exactly when the card has a UST cell): ffill levels, diffs, rolling std (300/60/20),
                     cumsum; None if the panel is absent.  _rates_row(c, *need): last row <= asof within 8 days, None if a needed asset is missing
 rates_feats(c, ns)  UST_10Y (u10_n) and all rates assets (lvpm) from the panel                                              [step 7: F1 u10_750, F2 u10_500, F3 u10_750 + lvpm]
 zlb_w(c)            UST_2Y and UST_10Y levels                                                                              [step 0, all families, UST level cards]
 zs_2y(c)            UST_2Y level                                                                                           [F4 f4s_z2, cards with a UST_2Y level cell]
 panel_feats(c)      all six tenors (UST_2Y 5Y 7Y 10Y 20Y 30Y) from the panel                                               [F2 / F3 wild; F3 PDX also uses only the card's own FX cell names]
 No history: f4s_w10, f4s_ros, fx_cap, CV, DP, ksl, KM, joint (uses only c['base'], c['drift'], c['anchor'] and the transformed X), the PDX FX sign table.
"""
import re
import numpy as np
import pandas as pd
RO = {'MKT': -1, 'SMB': -1, 'QMJ': 1, 'AUD': -1, 'NZD': -1, 'EUR': -1, 'GBP': -1, 'CAD': 1, 'NOK': 1, 'SEK': 1, 'DKK': 1, 'JPY': -1, 'CHF': -1}


NORM = {'lv20': (-0.115, 0.33), 'lv60': (-0.06, 0.25), 'lvm': (-0.09, 0.28), 'lt60': (-0.22, 0.68)}
TAU_GRID = np.exp(np.linspace(np.log(0.4), np.log(2.5), 25))
TEN = {'UST_2Y': 2.0, 'UST_5Y': 5.0, 'UST_7Y': 7.0, 'UST_10Y': 10.0, 'UST_20Y': 20.0, 'UST_30Y': 30.0}
TERM_NORM = {'ldzc': (-0.7258, 0.6331), 'cabs': (0.0, 0.2654), 'lvlN': (0.0, 0.3095), 'lvl': (0.0552, 0.3201)}
NMAX = 3300
C3 = 0.5
ZA0, ZA1, ZS0, ZS1 = 0.75, 1.75, 2.2, 1.2
ZS_Z0, ZS_Z1 = 1.0, 3.0
PAIR_LOGR, PAIR_GAP = 1.75, 3.0
FAC = ('MKT', 'SMB', 'HML', 'MOM', 'QMJ', 'BAB')
FXSET = ('AUD', 'NZD', 'EUR', 'GBP', 'CAD', 'NOK', 'SEK', 'DKK', 'JPY', 'CHF')
USDS = {'AUD': -1.0, 'EUR': -1.0, 'GBP': -1.0, 'NZD': -1.0, 'CAD': 1.0, 'CHF': 1.0, 'DKK': 1.0, 'JPY': 1.0, 'NOK': 1.0, 'SEK': 1.0}
PDN = ('f2m_60', 0.27, 0.88)


def _fam(**kw):
    q = dict(k=0.5, c=1.0, ct=1.3, ct2=1.3, beta=0.25, vlo=0.7, vhi=1.4, a=0.0, b=0.0, slope=0.0,
             rule=('lvm', 0.0, 0.0, 0.1, 0.0), shiftadj=False,
             G0=1.0, PW=0.5, CAP=2.5, gslope=0.0, rtail=0.0, ltail=0.0, LAM=0.5, TEXP=0.5,
             wv_ldzc=0.0, a_cabs=0.0, a_lvlN=0.0, wv_lvlN=0.0, m_lvl=0.0, xs=())
    q.update(kw)
    return q


# v10 base (per-family P: reshape / reversal constants; CFG: v10 add-on components)
P0 = {
    'F1': _fam(k=0.625, c=0.5, ct=1.0, ct2=1.15, a=0.1, b=-0.05, slope=0.05, rule=('lvm', 0.069, 0.0645, 0.1, 0.0),
               G0=0.75, ltail=0.3, LAM=0.3, wv_ldzc=0.1, a_cabs=0.0625, xs=(('u10', 750, -0.07),)),
    'F2': _fam(k=0.5, c=1.0, ct=1.3, ct2=1.3, rule=('lvm', -0.026, 0.178, 0.1, 0.0),
               G0=1.0, rtail=0.5, a_lvlN=0.3, wv_lvlN=0.125, xs=(('u10', 500, -0.12),)),
    'F3': _fam(k=0.375, c=0.9, ct=1.3, ct2=1.3, rule=('lvm', 0.094, 0.0, 0.1, 0.0),
               G0=1.0, rtail=0.5, gslope=0.3, xs=(('u10', 750, -0.135), ('lvpm', None, 0.36))),
    'F4': _fam(k=0.625, c=1.6, ct=1.5, ct2=1.7, beta=0.4375, a=0.3, b=0.25, slope=0.106, rule=('lv60', 0.0608, 0.3735, 0.1, 0.5),
               shiftadj=True, G0=1.4, PW=0.0, CAP=4.0, gslope=0.3, rtail=0.5, m_lvl=0.1),
}
CFG0 = {
    'F1': dict(CV=0.2, fx_cap=1.0),
    'F2': dict(ksl=0.2, DP=(0.75, 0.02, 0.09)),
    'F3': dict(ksl=0.3, lamS=0.5, DP=(0.75, 0.02, 0.09)),
    'F4': dict(CV=0.2, KM=0.7),
}

# round-4 components: each is (P overrides, CFG overrides) applied on top of the v10 base
COMPONENTS = {
    'f4s_w10': ({}, dict(W10=-0.12)),
    'f4s_r10': ({}, dict(R10=-0.15)),
    'f4s_z2': ({}, dict(Z2=-0.4)),
    'f4s_lv5': ({}, dict(LV5=-0.2)),
    'f4s_ros': ({}, dict(ROS={'JPY': 0.5, 'MKT': 1.5})),
    'wild_F2': ({}, dict(PD=[PDN + (0.15,)], PDH=0.5, PDS=0.3)),
    'wild_F3': ({}, dict(PD=[PDN + (0.10,)], PDH=0.5, PDS=0.3, PDX=0.15)),
    'f2r_rev': (dict(PW=0.35, rtail=0.75), {}),
    'f2r_fx': ({}, dict(fx_cap=0.75, fxm=(5, 0.1, 2.5))),
    'f2r_dt': ({}, dict(DP=(0.75, 0.02, 0.15))),
    'vol4': ({}, dict(VOL=dict(kap=4.0, b=0.25, lt=0.15))),
    'joint': ({}, dict(JOINT=dict(THETA=0.5, LAM_F1=0.3, LAM_OTHER=0.5))),
}
SELECTED = {
    'F1': ['joint'],
    'F2': ['wild_F2', 'f2r_rev', 'f2r_fx', 'f2r_dt'],
    'F3': ['wild_F3', 'vol4'],
    'F4': ['f4s_w10', 'f4s_r10', 'f4s_z2', 'f4s_lv5', 'f4s_ros', 'joint'],
}
P, CFG = {}, {}


def configure(selected=None, overrides=None):
    """rebuild P / CFG from the v10 base plus the selected components (overrides: {fam: {key: value}} patched into CFG[fam], {fam: {'P': {...}}} into P[fam])"""
    global SELECTED
    if selected is not None:
        SELECTED = {f: list(selected.get(f, [])) for f in P0}
    for f in P0:
        P[f] = dict(P0[f]); CFG[f] = dict(CFG0[f])
        for name in SELECTED[f]:
            dp, dc = COMPONENTS[name]
            P[f].update(dp); CFG[f].update(dc)
    for f, ov in (overrides or {}).items():
        ov = dict(ov)
        P[f].update(ov.pop('P', {})); CFG[f].update(ov)


# ---------------------------------------------------------------- as-of histories (the only access to SERIES)
_ST = {}
_MA = {}
_VH = {}
_OS = {}
_EW = {}
_RP = {}


def clear_caches():
    for d in (_ST, _MA, _VH, _OS, _EW, _RP):
        d.clear()


def hist(c, a, n=None):
    """levels of asset a (from the unit's own panel that holds a) at dates <= c['asof']; the last n if given"""
    s = c['H'](a)
    return s.iloc[-n:] if n else s


def _steps(s):
    dt = s.index.to_series().diff().dt.days
    hole = max(10 * dt.iloc[1:].median(), 5)
    st = s.diff()
    st[dt > hole] = np.nan
    return st.iloc[1:]


def steps_of(c, a):
    """last <= 3300 clean step changes (level diffs) of asset a up to asof"""
    key = (a, c['asof'])
    if key not in _ST:
        _ST[key] = _steps(hist(c, a, NMAX + 60)).dropna().to_numpy()[-NMAX:]
    return _ST[key]


def ma1500(c):
    """per-cell mean of the last 1500 levels up to asof of the cell's own asset (nan if < 500 points)"""
    key = (c['unit'], c['asof'])
    if key not in _MA:
        out = []
        for a, h in c['cells']:
            past = hist(c, a, 1500)
            out.append(past.mean() if len(past) >= 500 else np.nan)
        _MA[key] = np.array(out)
    return _MA[key]


def lvl_of(c, j):
    """log(sd of last 1500 steps / sd of last 300 steps) of cell j's own asset (0 if < 750 steps)"""
    a, h = c['cells'][j]
    key = (c['unit'], a, c['asof'])
    if key not in _VH:
        s = hist(c, a, 1700)
        dt = s.index.to_series().diff().dt.days
        hole = max(10 * dt.iloc[1:].median(), 5)
        st = s.diff() if c['ttype'] == 'level' else s.copy()
        st[dt > hole] = np.nan
        st = st.iloc[1:].dropna().to_numpy()
        _VH[key] = np.log(max(st[-1500:].std(), 1e-300) / max(st[-300:].std(), 1e-300)) if len(st) >= 750 else 0.0
    return _VH[key]


def own_steps(c, a):
    """last <= 1500 raw steps (level cells: np.diff of the last 1700 levels; log_return cells: the values) of asset a up to asof"""
    key = (c['unit'], a, c['asof'])
    if key not in _OS:
        x = hist(c, a, 1700).to_numpy()
        _OS[key] = (np.diff(x) if c['ttype'] == 'level' else x)[-1500:]
    return _OS[key]


def r250_of(c, a):
    """clip(sum of last 250 own steps / (sd of last 300 * sqrt 250), +-3); 0 if < 300 steps"""
    st = own_steps(c, a)
    v300 = st[-300:].std()
    if v300 <= 0 or len(st) < 300:
        return 0.0
    return float(np.clip(st[-250:].sum() / (v300 * np.sqrt(250)), -3, 3))


def lv5_of(c, a):
    """clip(log(rms of last 5 own steps / sd of last 300)/.55, +-3); 0 if < 300 steps"""
    st = own_steps(c, a)
    v300 = st[-300:].std()
    if v300 <= 0 or len(st) < 300:
        return 0.0
    return float(np.clip(np.log(max(np.sqrt((st[-5:] ** 2).mean()), 1e-12) / v300) / 0.55, -3, 3))


def ew_log_ratio(c, j, kap):
    """(log(ewma rms / sd300), log(sd1500 / sd300)) of cell j's own last <= 3300 clean steps up to asof; ewma lam = clip(1-kap/h, .85, .997); 0 where unavailable (sd1500 needs >= 1500 steps)"""
    a, h = c['cells'][j]
    key = (c['unit'], a, c['asof'], kap, h)
    if key not in _EW:
        s = hist(c, a, NMAX + 60)
        dt = s.index.to_series().diff().dt.days
        hole = max(10 * dt.iloc[1:].median(), 5)
        st = s.diff() if c['ttype'] == 'level' else s.copy()
        st[dt > hole] = np.nan
        st = st.iloc[1:].dropna().to_numpy()[-NMAX:]
        r = (0.0, 0.0)
        if len(st) >= 300:
            v3 = st[-300:].std()
            lam = float(np.clip(1.0 - kap / float(h), 0.85, 0.997))
            w = lam ** np.arange(len(st))[::-1]
            ew = np.sqrt((w * st * st).sum() / w.sum())
            v15 = st[-1500:].std() if len(st) >= 1500 else 0.0
            r = (float(np.log(ew / v3)) if (v3 > 0 and ew > 0) else 0.0, float(np.log(v15 / v3)) if (v3 > 0 and v15 > 0) else 0.0)
        _EW[key] = r
    return _EW[key]


# ---------------------------------------------------------------- rates panel helpers (all six UST tenors of the unit's rates_daily panel)
def _rates_panel(c=None):
    """all assets of the unit's rates panel up to as-of (None if not available): ffill levels, daily diffs, rolling stds, cumsums, levels"""
    if 'p' not in _RP:
        R = (c or {}).get('rates')
        if R is None or R.empty:
            _RP['p'] = None
        else:
            Lv = R.sort_index().ffill()
            W = Lv.diff().iloc[1:].fillna(0.0)
            _RP['p'] = dict(dates=W.index.values, cols=list(W.columns), cs=W.cumsum().values, lv=Lv.iloc[1:].values, vol=W.rolling(300, min_periods=200).std().values,
                            v20=W.rolling(20, min_periods=15).std().values, v60=W.rolling(60, min_periods=45).std().values)
    return _RP['p']


def _rates_row(c, *need):
    """index of the last rates-panel row <= asof if it is within 8 days and the panel has all the assets in need, else None"""
    Pn = _rates_panel(c)
    if Pn is None or any(a not in Pn['cols'] for a in need):
        return None
    i = int(np.searchsorted(Pn['dates'], np.datetime64(c['asof']), side='right')) - 1
    if i >= 0 and (np.datetime64(c['asof']) - Pn['dates'][i]) <= np.timedelta64(8, 'D'):
        return i
    return None


def rates_feats(c, ns):
    """{'u10_n': std. change of UST_10Y over last n steps, 'lvpm': pooled 0.5*(log v20/v300 + log v60/v300) over all rates assets} as of asof; {} if not available"""
    key = ('f', c['asof'])
    if key not in _RP:
        Pn = _rates_panel(c)
        i = _rates_row(c, 'UST_10Y')
        out = {}
        if i is not None:
            v = Pn['vol'][i]; ok = v > 0; j10 = Pn['cols'].index('UST_10Y')
            for n in ns:
                if i >= n + 1:
                    out[f'u10_{n}'] = float((Pn['cs'][i, j10] - Pn['cs'][i - n, j10]) / (v[j10] * np.sqrt(n))) if v[j10] > 0 else 0.0
            lv = [np.where(ok & (arr[i] > 0), np.log(np.where(arr[i] > 0, arr[i], 1.0) / np.where(ok, v, 1.0)), 0.0) for arr in (Pn['v20'], Pn['v60'])]
            out['lvpm'] = float(np.mean(0.5 * (lv[0] + lv[1])))
        _RP[key] = out
    return _RP[key]


def _ten(a):
    """tenor in years of UST asset id a: the TEN table, else parsed from UST_<n>Y / UST_<n>M, clipped to the calibrated 2-30y range (5y if unparseable)"""
    if a in TEN:
        return TEN[a]
    m = re.fullmatch(r'UST_?(\d+(?:\.\d+)?)([YM])', str(a).upper())
    t = (float(m.group(1)) / (12.0 if m.group(2) == 'M' else 1.0)) if m else 5.0
    return min(max(t, 2.0), 30.0)


def _ss(u):
    u = min(max(u, 0.0), 1.0)
    return u * u * (3.0 - 2.0 * u)


def zlb_w(c):
    """low-rate steep-curve weight in [0,1] from UST_2Y and UST_10Y levels at asof (0 if the rates panel is not available)"""
    key = ('w', c['asof'])
    if key not in _RP:
        Pn = _rates_panel(c)
        i = _rates_row(c, 'UST_2Y', 'UST_10Y')
        w = 0.0
        if i is not None:
            y2 = Pn['lv'][i, Pn['cols'].index('UST_2Y')]; y10 = Pn['lv'][i, Pn['cols'].index('UST_10Y')]
            if np.isfinite(y2) and np.isfinite(y10):
                w = (1.0 - _ss((y2 - ZA0) / (ZA1 - ZA0))) * (1.0 - _ss((ZS0 - (y10 - y2)) / (ZS0 - ZS1)))
        _RP[key] = float(w)
    return _RP[key]


def zs_2y(c):
    """smoothstep((UST_2Y level - 1)/2) at asof (0 if the rates panel is not available)"""
    key = ('z2', c['asof'])
    if key not in _RP:
        Pn = _rates_panel(c)
        i = _rates_row(c, 'UST_2Y')
        z = 0.0
        if i is not None:
            y2 = Pn['lv'][i, Pn['cols'].index('UST_2Y')]
            if np.isfinite(y2):
                z = _ss((y2 - ZS_Z0) / (ZS_Z1 - ZS_Z0))
        _RP[key] = float(z)
    return _RP[key]


def panel_feats(c):
    """UST curve front-end momentum at asof: f2m_60 = r_2Y(60) - mean(r_T(60) over the other five tenors), r_T(n) = (level_T[asof] - level_T[asof-n]) / (sd300_T * sqrt n)
    (sd300 = 300-row rolling std of daily changes).  {} if the rates panel row is missing / < 600 rows of history / any of the six tenors missing / a tenor's sd300 <= 0"""
    key = ('pf', c['asof'])
    if key not in _RP:
        i = _rates_row(c, *TEN)
        out = {}
        if i is not None and i >= 600:
            Pn = _rates_panel(c)
            cols = [Pn['cols'].index(a) for a in TEN]
            v = Pn['vol'][i, cols]
            if (v > 0).all():
                rr = (Pn['cs'][i, cols] - Pn['cs'][i - 60, cols]) / (v * np.sqrt(60.0))
                out['f2m_60'] = float(rr[0] - rr[1:].mean())
        _RP[key] = out
    return _RP[key]


# ---------------------------------------------------------------- component 2: UST prior-window reversal
def reversal(c, sd, q, w=0.0, lamS=0.0):
    """(drift after reversal, correction in horizon-sd units cr = -corr/sd)"""
    d = np.array(c['drift'], float).copy()
    cr = np.zeros(len(d))
    if c['ttype'] != 'level':
        return d, cr
    mv = None
    if lamS and len({a for a, h in c['cells'] if a.startswith('UST')}) >= 2:
        mm = {}
        for a in TEN:
            try:
                st = steps_of(c, a)
            except KeyError:
                continue
            if len(st) >= 900:
                mm[a] = float(st[-600:-300].mean() + C3 * st[-900:-600].mean())
        if len(mm) >= 3:
            xs_ = np.log(np.array([TEN[a] for a in mm])); vals = np.array(list(mm.values()))
            xm = xs_.mean(); xc = xs_ - xm; den = (xc ** 2).sum()
            b0 = (xc * (vals - vals.mean())).sum() / den if den > 0 else 0.0
            mv = (vals.mean(), b0, xm)
    for j, (a, h) in enumerate(c['cells']):
        if a.startswith('UST'):
            st = steps_of(c, a)
            if len(st) >= 900:
                g = q['G0'] * (21.0 / h) ** q['PW']
                if q['gslope']:
                    g *= max(1.0 + q['gslope'] * np.log(10.0 / _ten(a)) / np.log(5.0), 0.0)
                g *= 1.0 - w
                cap = q['CAP'] * sd[j]
                mj = st[-600:-300].mean() + C3 * st[-900:-600].mean()
                if mv is not None:
                    a0_, b0_, xm_ = mv
                    xj = np.log(_ten(a)) - xm_
                    mj = a0_ + (h / 63.0) ** lamS * b0_ * xj + (mj - a0_ - b0_ * xj)
                corr = float(np.clip(g * h * mj, -cap, cap))
                d[j] -= corr
                cr[j] = -corr / sd[j]
    return d, cr


def xasset_shift(c, q, sd, isU):
    xs = np.zeros(len(isU))
    if not q['xs']:
        return xs
    f = rates_feats(c, (500, 750))
    for j in np.where(isU > 0)[0]:
        v = 0.0
        for name, n, coef in q['xs']:
            if name == 'u10':
                x = f.get(f'u10_{n}')
            else:
                x = f.get('lvpm')
                if x is not None:
                    ds = c['drift'][j] / sd[j]
                    x = x * ds / (abs(ds) + 0.1)
            if x is not None and np.isfinite(x):
                v += coef * float(np.clip(x, -3.0, 3.0))
        xs[j] = v * (c['hs'][j] / 63.0) ** 0.5
    return xs


def pd_shift(c, spec, isU, hexp=0.0, ten_s=0.0):
    """sum over (feat, cen, sc, coef) of coef * clip((f - cen)/sc, +-2.5) on UST level cells (feats from panel_feats)"""
    xs = np.zeros(len(isU))
    if not spec or not (isU > 0).any():
        return xs
    pf = panel_feats(c)
    v = 0.0
    for name, cen, sc, coef in spec:
        f = pf.get(name)
        if f is not None and np.isfinite(f):
            v += coef * float(np.clip((f - cen) / sc, -2.5, 2.5))
    xs[isU > 0] = v
    if hexp:
        xs = xs * (np.asarray(c['hs'], float) / 63.0) ** hexp
    if ten_s:
        xs = xs * np.array([max(1.0 + ten_s * np.log(10.0 / TEN[a]) / np.log(5.0), 0.0) if a in TEN else 1.0 for a, h in c['cells']])
    return xs


def vol_factor(c, q, feat, vc):
    if vc is None:
        return np.clip(np.array([f['v20'] / f['v300'] if f['v300'] > 0 else 1 for f in feat]) ** q['beta'], q['vlo'], q['vhi'])
    g = []
    for j in range(len(feat)):
        e = ew_log_ratio(c, j, vc['kap'])
        g.append(vc['b'] * e[0] + vc['lt'] * e[1])
    return np.clip(np.exp(np.array(g)), q['vlo'], q['vhi'])


# ---------------------------------------------------------------- pair step (d == 2) : exact-variogram rank coupling
def _couple(Xa, Xb, za, zb, rho):
    sa = za - za.mean(); sa = sa / sa.std()
    eb = zb - zb.mean(); eb = eb - (eb @ sa) / (sa @ sa) * sa
    sd_ = eb.std()
    eb = eb / sd_ if sd_ > 1e-12 else eb
    wv = rho * sa + np.sqrt(max(1 - rho * rho, 0.0)) * eb
    order = np.argsort(wv, kind='stable')
    out = np.empty_like(Xb)
    out[order] = np.sort(Xb)
    return out


def match_pair(Xa, Xb, za, zb, target, power=0.5, iters=40):
    """rho in [-.999,.999] (bisection) such that mean|Xb'-Xa|^.5 = target, Xb' = sorted Xb re-paired to Xa by a Gaussian copula in M0's normal scores; (Xb', rho, status)"""
    f = lambda r: np.mean(np.abs(_couple(Xa, Xb, za, zb, r) - Xa) ** power)
    lo, hi = -0.999, 0.999
    flo, fhi = f(lo), f(hi)
    if target >= flo:
        return _couple(Xa, Xb, za, zb, lo), lo, 'lo'
    if target <= fhi:
        return _couple(Xa, Xb, za, zb, hi), hi, 'hi'
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if f(mid) > target:
            lo = mid
        else:
            hi = mid
    r = 0.5 * (lo + hi)
    return _couple(Xa, Xb, za, zb, r), r, 'ok'


def fix_scale(X0, Xb, mm, LAM):
    Dp = Xb - X0; mp = np.mean(np.abs(Dp) ** .5)
    Dn = ((mm / mp) ** 2 if mp > 0 else 1.0) * Dp
    Ab = LAM * X0 + (1 - LAM) * Xb
    return np.column_stack([Ab - (1 - LAM) * Dn, Ab + LAM * Dn])


def fix_theta(X0, Xb, mm, LAM, theta, Dm0):
    """D' = al + s*(D - mean D), al = (1-theta)*mean(D) + theta*Dm0, s solves mean|D'|^.5 = mm (None if impossible)"""
    D = Xb - X0; mu = D.mean(); e = D - mu
    al = (1 - theta) * mu + theta * Dm0
    f = lambda s: np.mean(np.abs(al + s * e) ** .5)
    lo, hi = 1e-3, 1.0
    if f(hi) < mm:
        while f(hi) < mm and hi < 1e6:
            hi *= 2
    elif f(lo) > mm:
        return None
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if f(mid) < mm: lo = mid
        else: hi = mid
    s = 0.5 * (lo + hi)
    Dn = al + s * e
    Ab = LAM * X0 + (1 - LAM) * Xb
    return np.column_stack([Ab - (1 - LAM) * Dn, Ab + LAM * Dn])


def joint_pair(c, X, z, jc):
    m0 = c['anchor'] + c['drift'] + c['base']
    mm = np.mean(np.abs(m0[:, 1] - m0[:, 0]) ** 0.5)
    Xa = X[:, 0]
    Xb, rho, st = match_pair(Xa, X[:, 1], z[:, 0], z[:, 1], mm)
    if st == 'ok':
        Dp = Xb - Xa
        mp = np.mean(np.abs(Dp) ** 0.5)
        Dn = ((mm / mp) ** 2 if mp > 0 else 1.0) * Dp
        return np.column_stack([Xb - Dn, Xb])
    if c['fam'] == 'F1':
        Dm0 = float(np.mean(m0[:, 1] - m0[:, 0]))
        Y = fix_theta(Xa, Xb, mm, jc['LAM_F1'], jc['THETA'], Dm0)
        if Y is not None:
            return Y
        return fix_scale(Xa, Xb, mm, jc['LAM_F1'])
    return fix_scale(Xa, Xb, mm, jc['LAM_OTHER'])


# ---------------------------------------------------------------- transform
def _x(f, key):
    cen, sc = TERM_NORM[key]
    return np.clip((f - cen) / sc, -2.5, 2.5)


def transform_card(c):
    fam = c['fam']; q = P[fam]; cf = CFG[fam]
    base = c['base']; sd0 = base.std(0, keepdims=True); sd = np.where(sd0 > 0, sd0, 1)
    z = base / sd
    az = np.abs(z); sgn = np.sign(z)
    cells = c['cells']
    d = base.shape[1]
    feat = c['feat']
    dr0 = np.asarray(c['drift'], float)
    ros = cf.get('ROS', {})
    ro = np.array([RO.get(x, 0) * ros.get(x, 1.0) for x, h in cells], float)[None, :]
    isU = np.array([1.0 if (x.startswith('UST') and c['ttype'] == 'level') else 0.0 for x, h in cells])
    isr = isU > 0
    v300 = np.array([f['v300'] for f in feat]); ok = v300 > 0; v3 = np.where(ok, v300, 1.0)

    wz = zlb_w(c) if isr.any() else 0.0
    dadj, crs = reversal(c, sd[0], q, wz, cf.get('lamS', 0.0))                  # 2
    isf = np.array([1.0 if (x in FAC and c['ttype'] == 'log_return') else 0.0 for x, h in cells])
    isF = np.array([1.0 if x in FXSET else 0.0 for x, h in cells])

    m = 1 + q['a'] * sgn * ro                                                   # 1
    cm = 1.0
    if q['slope']:
        r60 = np.array([f['r60'] for f in feat])
        lt = np.log(np.where(ok, np.abs(r60) / (v3 * np.sqrt(60)), 0.0) + 0.15)
        cm = np.exp(q['slope'] * np.clip((lt - NORM['lt60'][0]) / NORM['lt60'][1], -2.5, 2.5))[None, :]
    if q['rtail'] and isr.any():                                                # 3
        m = m * (1 + q['rtail'] * sgn * np.clip(crs, -1, 1)[None, :])
    if q['ltail'] and isr.any():                                                # 4
        ma = ma1500(c)
        ma = np.where(np.isfinite(ma), ma, c['anchor'])
        m = m * (1 + q['ltail'] * (1.0 - wz) * sgn * np.where(isr, np.clip((ma - c['anchor']) / 3.0, -1, 1), 0.0)[None, :])

    fe, sa, sb, eps, gam = q['rule']                                            # 5
    lv20 = np.where(ok, np.log(np.maximum([f['v20'] for f in feat], 1e-300) / v3), 0.0)
    lv60 = np.where(ok, np.log(np.maximum([f['v60'] for f in feat], 1e-300) / v3), 0.0)
    lv = {'lv20': lv20, 'lv60': lv60, 'lvm': 0.5 * (lv20 + lv60)}[fe]
    dr = dadj if q['shiftadj'] else dr0
    shift = (sa + sb * np.clip((lv - NORM[fe][0]) / NORM[fe][1], -2.5, 2.5)) * dr / (np.abs(dr) + eps * sd[0] + 1e-300)
    hs_ = np.asarray(c['hs'], float)
    if gam:
        shift = shift * (hs_ / 63.0) ** gam

    lvm = np.zeros(d)                                                           # 6
    dirdr = dr0 / (np.abs(dr0) + 0.1 * sd[0] + 1e-300)
    if q['wv_ldzc']:
        lvm += q['wv_ldzc'] * _x(np.log(np.abs(dadj / sd[0]) + 0.15), 'ldzc')
    if q['a_cabs']:
        shift = shift + q['a_cabs'] * _x(np.abs(crs), 'cabs') * dirdr
    if q['a_lvlN'] or q['wv_lvlN'] or q['m_lvl']:
        lvl = np.array([lvl_of(c, j) for j in range(d)])
        if q['a_lvlN'] or q['wv_lvlN']:
            xN = _x(lvl * (1 - isU), 'lvlN')
            shift = shift + q['a_lvlN'] * xN * dirdr
            lvm += q['wv_lvlN'] * xN
        if q['m_lvl']:
            shift = shift + q['m_lvl'] * _x(lvl, 'lvl') * np.clip(dadj / sd[0], -3, 3)
    shift = shift + xasset_shift(c, q, sd[0], isU)                              # 7
    if cf.get('PD'):                                                            # 7c wild4
        shift = shift + pd_shift(c, cf['PD'], isU, cf.get('PDH', 0.0), cf.get('PDS', 0.0))
    if cf.get('PDX') and isr.any() and isF.any():
        xv = pd_shift(c, [(n_, cen_, sc_, 1.0) for n_, cen_, sc_, _ in cf['PD'][:1]], np.ones(d))
        usd = np.array([USDS.get(a, 0.0) for a, h in cells]) * (isF > 0)
        shift = shift + cf['PDX'] * usd * xv * (hs_ / 63.0) ** cf.get('PDH', 0.0)
    if cf.get('fx_cap') and c['ttype'] == 'level':                              # 7b
        for j in np.where(isF > 0)[0]:
            cp = cf['fx_cap'] * sd[0][j]
            shift[j] += q['k'] * (float(np.clip(dadj[j], -cp, cp)) - dadj[j]) / sd[0][j]
    if cf.get('fxm') and c['ttype'] == 'level':
        fn, fb, fcl = cf['fxm']
        for j in np.where(isF > 0)[0]:
            st = steps_of(c, cells[j][0])
            if len(st) >= 300 and st[-300:].std() > 0:
                shift[j] += fb * float(np.clip(st[-fn:].sum() / (st[-300:].std() * np.sqrt(fn)), -fcl, fcl)) * np.sqrt(hs_[j] / 63.0)
    if cf.get('CV') and isf.any():
        lvmf = np.clip(0.5 * (lv20 + lv60), -3.0, 3.0)
        shift = shift - cf['CV'] * isf * lvmf * np.sqrt(hs_ / 63.0)
    hp = np.sqrt(hs_ / 21.0)                                                    # 7d f4shock
    if cf.get('W10') and c['ttype'] == 'level':
        lvm = lvm + cf['W10'] * np.array([1.0 if x == 'UST_10Y' else 0.0 for x, h in cells])
    if cf.get('R10') and c['ttype'] == 'level':
        is10 = np.array([1.0 if x == 'UST_10Y' else 0.0 for x, h in cells])
        if is10.any():
            shift = shift + cf['R10'] * is10 * np.array([r250_of(c, a) if is10[j] else 0.0 for j, (a, h) in enumerate(cells)]) * hp
    if cf.get('Z2') and c['ttype'] == 'level':
        is2 = np.array([1.0 if x == 'UST_2Y' else 0.0 for x, h in cells])
        if is2.any():
            shift = shift + cf['Z2'] * is2 * zs_2y(c) * hp
    if cf.get('LV5') and isf.any():
        shift = shift + cf['LV5'] * isf * np.array([lv5_of(c, a) if isf[j] else 0.0 for j, (a, h) in enumerate(cells)])
    tailm = 1.0
    if cf.get('DP'):                                                            # D
        dcen, dw, dt = cf['DP']
        xd = np.clip((np.abs(dadj / sd[0]) / np.sqrt(hs_ / 63.0) - dcen) / 0.5, -2.5, 2.5)
        lvm = lvm + dw * xd
        tailm = (1 + dt * xd)[None, :]
    vm = np.exp(lvm)[None, :] if np.any(lvm) else 1.0

    zt = sgn * (q['c'] * cm * np.minimum(az, 1) + (q['ct'] * m * np.clip(az - 1, 0, 1) + q['ct2'] * m * np.maximum(az - 2, 0)) * tailm)
    vr = vol_factor(c, q, feat, cf.get('VOL'))
    centre = c['anchor'] + q['k'] * np.where(isf > 0, cf.get('KM', 1.0), 1.0) * dadj
    if cf.get('ksl') and isr.any():
        for j in np.where(isr)[0]:
            fj = max(1.0 + cf['ksl'] * np.log(5.0 / _ten(cells[j][0])) / np.log(2.5), 0.0)
            centre[j] += (fj - 1.0) * q['k'] * dr0[j]
    X = centre + sd * (zt * vr[None, :] * vm + q['b'] * ro + shift[None, :])
    if isr.any():                                                               # 8
        X = X.copy()
        for j in np.where(isr)[0]:
            a0 = c['anchor'][j]; L = max(a0 - 0.2, 0.05) * 2.0
            dl = X[:, j] - a0; neg = dl < 0
            X[neg, j] = a0 - L * np.tanh(-dl[neg] / L)

    m0 = c['anchor'] + c['drift'] + base                                        # 9
    if d == 2:
        same = cells[0][0] == cells[1][0]
        D0 = m0[:, 1] - m0[:, 0]
        if abs(np.log(sd[0, 1] / sd[0, 0])) > PAIR_LOGR:
            jd = 0 if sd[0, 0] <= sd[0, 1] else 1
            X = m0 + (X[:, jd:jd + 1] - m0[:, jd:jd + 1])
        elif cf.get('JOINT') and abs(D0.mean()) <= PAIR_GAP * max(D0.std(), 1e-300):
            X = joint_pair(c, X, z, cf['JOINT'])
        elif fam == 'F1' and same:
            Dp = X[:, 1] - X[:, 0]
            mp = np.mean(np.abs(Dp) ** 0.5); mm = np.mean(np.abs(m0[:, 1] - m0[:, 0]) ** 0.5)
            Dn = ((mm / mp) ** 2 if mp > 0 else 1.0) * Dp
            LAM = q['LAM']
            Ab = LAM * X[:, 0] + (1 - LAM) * X[:, 1]
            X = np.column_stack([Ab - (1 - LAM) * Dn, Ab + LAM * Dn])
        else:
            X = m0 + (X[:, :1] - m0[:, :1])
    elif fam == 'F3' and c['ttype'] == 'log_return' and d >= 3:
        ctr = centre[None, :]
        u = (X - ctr) / sd; ub = u.mean(1, keepdims=True); dv = u - ub
        iu = np.triu_indices(d, 1)
        t0 = (np.abs(m0[:, iu[0]] - m0[:, iu[1]]) ** 0.5).mean(0)
        errs = [(((np.abs((ctr + sd * (ub + t * dv))[:, iu[0]] - (ctr + sd * (ub + t * dv))[:, iu[1]]) ** 0.5).mean(0) - t0) ** 2).sum() for t in TAU_GRID]
        X = ctr + sd * (ub + TAU_GRID[int(np.argmin(errs))] ** q['TEXP'] * dv)
    return X


configure(SELECTED)

# v12 hedge (2026-10-03): the UST signal stack (reversal gain, tenor gradient, reversal-keyed tail, rates cross-asset
# u10 lean, front-end momentum PD, tenor drift gradient ksl, F4 10Y reversal R10 and 2Y-gated skew Z2) at 75% strength.
UST_STACK = 0.75
for _f in P:
    P[_f]['G0'] *= UST_STACK
    P[_f]['gslope'] *= UST_STACK
    P[_f]['rtail'] *= UST_STACK
    P[_f]['xs'] = tuple((nm, n, co * UST_STACK if nm == 'u10' else co) for nm, n, co in P[_f]['xs'])
for _f in CFG:
    if CFG[_f].get('PD'):
        CFG[_f]['PD'] = [(n_, cen_, sc_, co_ * UST_STACK) for n_, cen_, sc_, co_ in CFG[_f]['PD']]
    for _k in ('ksl', 'R10', 'Z2'):
        if _k in CFG[_f]:
            CFG[_f][_k] = CFG[_f][_k] * UST_STACK
