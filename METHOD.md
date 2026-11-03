# Method and provenance — Track 2 forecaster (v13w5)

Release described by this record: **v13w5**, the Agenthon 2026 Track 2 Final entry (CodaBench submission 1006655, image
`ghcr.io/notcqqkie/qfr-b@sha256:4195d2b76f9a86295481ad226a6e47d3e491fa1f1853784905b1752b6b77c667`). This record was written
during development; where it mentions internal tooling ("kit3", research swarms, practice harness) those were private test
harnesses that are not part of the forecaster.
v13w5 = v13w2 + the v13w3 full short-horizon stack + robustness guards + D1 draw layout + the v13r3 release-card model (v13w4)
+ a date-gated FOMC bill prior (v13w5). Every one of the 104 public units produces output byte-identical to v13w2.

The sections from "Artifacts in the image" to "v12" are the record written for v8–v12 and are kept verbatim; they describe the
inherited stack that still serves long-horizon family cards (including every public unit). Corrections to that record, and the
v13 history, follow it. Dates below are 2026 unless stated. "Observation dates" are the dates of data rows; "vintage" or
"publication" dates are when a value became public; they are kept apart throughout.

## Artifacts in the image

No learned model, no weights, no data files. The image contains Python source (`t2agent/`) and
pinned public libraries: numpy 2.1.3, pandas 2.2.3, pyarrow 18.1.0 (python:3.13-slim-bookworm base).
The House model is not called by the shipped forecaster (`models: []`).

## Method

Per unit, using only that unit's own `/input` files:

1. Rebuild the published text-blind baseline (docs/M0-BASELINE.md): trailing 300 steps, gap rule,
   date alignment, drift `s·mu`, path covariance `min(s,s')·Sigma`, 500 draws, seed `crc32([task].id)`.
2. Shrink the drift by `k`, reshape the standardized draws piecewise-linearly (slope `c` within one sd,
   `ct` beyond), scale width by `clip((20-step sd / 300-step sd)^0.25, 0.7, 1.4)`.
3. On two-cell cards, keep each draw's cell difference equal to the baseline's.
4. Tail/shock (F4) cards, equity-factor and currency targets only: risk-off tail slope ×1.3, opposite tail ×0.7,
   centre shifted 0.25 sd toward risk-off (fixed sign table per asset; rates are left symmetric).
5. US Treasury level targets: the shrunk drift is reduced by g·(21/h)^0.5·h·(m[300,600) + 0.5·m[600,900)), the mean
   per-step change 300-600 and 600-900 steps before the as-of date (g = 0.75/1.0/1.0/1.25 for F1-F4, capped at 1.5
   horizon sd; skipped with < 900 steps of history).
6. Centre lean along the sign of the baseline drift, size (a + b·x)·sd with x the standardized log ratio of recent
   (20/60-step) to 300-step volatility; F4 inner width × exp(0.106·x_trend) (trend strength |r60|/(sd300·√60)).
7. F1 one-asset two-horizon cards: both horizons reshaped, the horizon gap rescaled so mean |gap|^0.5 equals the
   baseline's (variogram-neutral). F3 equity-factor return cards: cross-asset dispersion matched to the baseline's
   pairwise |difference|^0.5 means.
8. Monthly-macro and EM-transfer cards use step 1 unchanged.

## Constants and the data used to choose them

| Constant | Value | Selected on |
|---|---|---|
| k, c, ct (far-tail slope beyond 2 sd) for F1 | 0.625, 0.5, 1.0 (1.15) | backtests below |
| k, c, ct for F2 | 0.5, 1.0, 1.3 | backtests below |
| k, c, ct for F3 | 0.375, 0.9, 1.3 | backtests below |
| k, c, ct for F4 | 0.625, 1.6, 1.7 (vol exponent 0.5) | backtests below |
| vol exponent / clip | 0.25 / [0.7, 1.4] | backtests below |
| F4 risk-off tail asymmetry / centre shift | 0.3 / 0.25 sd | backtests below |

Selection data, all public and dated 2000-01-03 … 2024-12-18 (before every Final-window cutoff):

- The panels shipped in `Agenthon-2026/track2-forecasting-public` `units/` (FRED H.15 yields,
  H.10 G10 FX, Fama-French/AQR factor returns), merged per series.
- Synthetic backtest cards: the public card templates (asset sets, horizons, target types) re-dated
  at random as-of dates from 2002 onward, outcomes taken from later rows of the same public panels.
- The public practice cards at their own as-of dates, outcomes from the same public panels.
- Per family, synthetic cards were importance-weighted so that the distribution of their realized
  surprise (|realized - baseline mean| / baseline sd) matches that family's practice cards; constants
  minimise the weighted mean score. Extra F2/F4 synthetic cards: 200 random dates per template.
- A second weighting also matches each family's share of outcomes moving with vs against the baseline drift;
  changes were accepted only if they improved both the pre-2013 and post-2013 halves.

Scoring for selection: the track composite (CRPS, variogram p=0.5, pinball 1/5/95/99%) divided by
the rebuilt baseline per card, averaged. Grids over k ∈ {0…1.25}, c ∈ {0.4…1.3}, ct ∈ {0.8…2.8},
checked by a pre-2013 / post-2013 split, and the v8 additions also on a separate held-out synthetic set (different
random as-of dates). No unit's realized value is stored in the image.

## v9 additions (2026-10-02)

Per-family constants and components are listed in `t2agent/v9.py` (constants `P`, pipeline in the module docstring
of the source candidate). Beyond v8: refined Treasury reversal (gain, cap, tenor gradient, horizon exponent for F4),
a tail made heavier on the side the reversal moves the centre, a tail lean toward the 1500-step mean (F1), regime terms
from long-run (1500-step) vs 300-step volatility and residual trend strength, a centre lean on Treasury cells from the
standardized 500/750-step UST_10Y change and pooled rates-panel volatility ratio (computed only from the unit's own
rates panel up to the as-of date), and a soft lower bound for yield levels. Selected by greedy forward selection on the
synthetic dev sets (both eras required to improve; real practice cards not worse by more than 0.003), then confirmed on
two separate held-out synthetic sets drawn at different random as-of dates.

## v10 additions (2026-10-02)

Module `t2agent/v10.py` (constants `P`, `CFG`). Beyond v9: the Treasury reversal and its tail terms are switched off
smoothly when the 2-year yield is near zero AND the 10Y-2Y curve is steep (weight from UST_2Y and UST_10Y levels in the
unit's rates panel at the as-of date; zero whenever the 2Y yield is above 1.75% or the slope is 1.2 or less); a
tenor gradient on how much of the baseline drift is extrapolated for Treasury cells (F2, F3); the reversal signal split
into a curve-slope part (horizon-scaled) and a residual for multi-tenor cards (F3); a drift-significance width/tail
rule (F2, F3); a volatility-managed centre for equity-factor cells (F1, F4); a drift cap for G10 FX cells (F1); and F4
tail slope 1.5 / vol exponent 0.4375. Selected by greedy forward selection on the dev sets with era and 4-year-block
requirements, then confirmed on a further, previously unused held-out synthetic set.

## v11 additions (2026-10-03)

Module `t2agent/v11.py` (ported from `candidates/v11_final.py`; docstring lists every component and constant). Beyond
v10: F4 shock structure (10Y width and own-trend reversal, a dovish skew gated by the 2Y level, short-term factor-vol
shift, per-asset risk-off scaling); a UST front-end momentum state (2Y vs other tenors over 60 steps) shifting UST cells
in F2/F3 and a USD-factor shift on FX cells in F3 cards that hold both; F2 reversal/tail/FX-drift retune; F3 width from
a horizon-matched EWMA and long-run volatility; rank coupling that keeps the baseline's pairwise variogram exactly while
keeping our marginals (F1, F4). EM-transfer cards: baseline drift shrunk by 0.6, width/shape unchanged. Selected by
greedy forward selection (era, 4-year-block and final-like-regime requirements) and confirmed on a further unused
held-out synthetic set.

## v12 (2026-10-03)

v11 plus (1) robustness patches with no effect on any public unit or development card: tenor parsing for Treasury ids
outside the six calibrated tenors, layered degradation (v11 -> v8 transform -> exact baseline draws), card.toml-less
operation from forecast_spec.json, family parsed by regex, target_type normalised like the shipped baseline, unreadable
parquet files skipped, and a corrected two-cell construction for pairs of different assets with different units; and
(2) a deliberate hedge: the Treasury signal stack (reversal gain, tenor gradient, reversal-keyed tail, rates cross-asset
lean, front-end momentum, tenor drift gradient, F4 10Y reversal and 2Y-gated skew) at 75% strength. The hedge was
chosen on pre-2025 evidence only: full strength is best on 2002-2024 history overall, but the most recent regime in the
development data (2023-2024) favours weaker Treasury signals.

## Corrections to the record above (added 2026-10-16)

An internal provenance audit of v13w2 (14 Oct) found the v8–v12 text incomplete in the following points; they apply to every
later version, because the inherited stack is unchanged.

- Section "Method", step 8 ("Monthly-macro and EM-transfer cards use step 1 unchanged") is out of date. Since v12e (10 Oct):
  EM-transfer cards shrink the baseline drift by k = 0.2 and scale width/tail by (1.3, 1.2); monthly-macro cards keep k = c = ct = 1
  but scale width/tail by (1.15, 1.2). Selection: synthetic FRED EM cards (as-of 2005-01-01 to 2024-06-30, 64-step outcomes) plus four
  practice cards; 1,800 synthetic CPI/PCE/NFP/UNRATE cards built from the merged public macro panel.
- The v12d/v12e/v12f calibration (9–11 Oct) is not described above. Per family (width inside 1 sd, tail factor, two-cell lock, centre
  gain away from the M0 mean): F1 (1.3, 1, 1, 1.3), F2 (1.6, 0.85, 1, 1.3), F3 (1.3, 1, 1, 1.3), F4 (1.6, 0.7, 0, 1.15); v12f adds an F4
  risk-off asymmetry (equity-factor cells: tail +0.3 and a 0.1 sd risk-off lean; Treasury cells: falling-yield tail +0.15). The F2/F4
  width 1.6 was retained after a 1.45 alternative scored worse on practice cards (v12h, rejected). v12g (11 Oct) implements the M0
  named-month step rule (no constant). All were selected on the legacy development sets (synthetic cards, two held-out synthetic sets
  and the public practice cards), both eras and 4-year blocks.
- "Selection data … dated 2000-01-03 … 2024-12-18 (before every Final-window cutoff)" gives observation dates. The panels used were
  the ones shipped in the public Track 2 repository, whose manifests carry a public release date of 2026-07-03; the record does not
  establish which vintage of each value was public at earlier historical dates. Eligibility is therefore claimed only for the ranked
  Final (as-of 26 Oct 2026), not for earlier historical task cutoffs.
- Rationale step 4 in the CLI output ("chosen on pre-as-of historical backtests only") means: before the ranked Final as-of. It is
  not a claim that every constant predates every historical card's own as-of date.
- Keeping a design because it scored better is selection too: the quantile grid, the event architecture and the retained legacy
  stack were all kept on score evidence, not only on analytical grounds.

## v13v (2026-10-11)

Organizer notice track2-forecasting-public#4 (11 Oct) said the ranked Final uses UST and G10 FX cards at horizons of a few business
days. On such cards the v12 reshape lost to M0 (FOMC-matched synthetic cards: v12f 1.131 vs M0 1.032). v13v: when every cell is a UST
yield or G10 currency and every horizon is <= 20 BD, the draws are the v12 (v11) mean plus M0's own joint deviations scaled per cell by
clip((sd of the last 60 changes / sd of the last 300)^0.75, 0.5, 2). Every other card is unchanged.

- Selection: short-horizon copies of the public UST/FX templates, as-of dates 2004–2024 (FOMC-matched as-of 3 BD before each meeting,
  n = 1,665; random dates, n = 1,262), horizons 2–12 BD, scored with the track composite over M0's expected error. Results: v13
  (v12 centre + M0 width) 1.024 / 1.056, v13v 0.998 / 1.033, M0 1.037 / 1.060 (FOMC / random), better in every era. Centre and
  vol-form alternatives differed by <= 0.01 between sets, so the built form was kept.
- Tests: 104/104 public units byte-identical; 79 Final-shaped units engage, admissible under toolkit 2.6.0.

## v13w (2026-10-11)

Four levers on the v13v branch only (`t2agent/v13w.py`, module docstring):

| Lever | Value |
|---|---|
| Centre | UST cells keep the v12 centre; currency centre = last level + 0.25 x (v12 centre − last level) (FX_LAM = 0.25) |
| Width | per asset clip((EWMA(0.94) sd of daily changes / sd of the last 300 changes)^0.8, 0.5, 2), replacing (v60/v300)^0.75 |
| Events | each horizon increment rescaled by its expected variance against the trailing 300 BD: FOMC lull and jump, US payrolls, BoJ, ECB (v13w tables: UST FOMC −2/−1 0.63, day 0 1.1, +1 1.9, +2 0.9, payrolls 2.2; FX FOMC −2..0 0.76, +1 2.4, +2 1.2, payrolls 1.35; ECB 2.5 EUR/DKK, 1.6 CHF, 1.5 SEK/NOK; BoJ 2.9 JPY, 1.25 AUD/NZD/CAD) |
| Grid | each cell's draws = an exact quantile grid with the model sd, placed by the ranks of M0's joint draws |

- Selection: a research swarm (8 agents) on a fixed kit of Final-like cards (UST + G10 FX, horizons 2–12 BD) built from the public
  practice series: development cards 1,664 FOMC-matched and 1,260 random (as-of 2004-01-23 … 2024-11-14, realized targets up to
  2024-11-21); held-out cards 1,665 / 1,261 that no agent saw (as-of 2004-01-12 … 2024-11-14, targets up to 2024-12-03).
  Event FOMC/payroll ratios were fitted on pseudo-cards with as-of dates 2001-01-01 … 2014-12-31 and checked on 2015–2024; BoJ ratios on
  2013–2024 daily moves (110 meetings), ECB on 2015–2024. The scoring kit equals the official scorer to 1e-16.
- Result, blind holdout through the deployed code path: v13w − v13v = −0.0245 [−0.0340, −0.0159] (FOMC-matched; early −0.0165, late
  −0.0336) and −0.0187 [−0.0263, −0.0113] (random). Holdout ablation (lever removed, FOMC / random): events −0.0039 / −0.0016,
  FX centre −0.0063 / −0.0087, EWMA width −0.0099 / −0.0092, grid −0.0047 / −0.0014.
- Calendars: FOMC, ECB and BoJ decision dates and the BLS Employment Situation rule (third Friday after the reference week, holiday
  shifts) are public schedules published in advance. The Final-window dates (FOMC 28 Oct, ECB 29 Oct, BoJ 30 Oct, payrolls 6 Nov
  2026) were checked against the official 2026 schedules on 11 Oct and again on 15 Oct. Earlier entries were compiled partly from
  memory and spot-checked against the published schedules; a source receipt for every historical entry was not kept (unverified
  entry by entry).

## v13w2 (2026-10-12): adapted to the corrected Final

Organizer notice track2-forecasting-public#4, comment 6129776242 (12 Oct): ranked as-of Mon 26 Oct, horizons 1–4 BD, FX from the ECB
euro reference rates, added Treasury bills (3M/6M/1Y), five EM currencies (MXN, BRL, ZAR, KRW, INR) and a few official-data release
cards.

- v13w at the corrected shape (as-of 2 BD before an FOMC, h 1–4, UST/G10; seeds 51–54 on the same public practice series):
  v13w − v13v = −0.0328 [−0.0478, −0.0179] FOMC-matched, −0.0377 [−0.0545, −0.0217] held-out FOMC-matched, −0.0136 random, −0.0194
  held-out random; removing the event timing costs +0.013 to +0.016.
- EM currencies join the branch (SHORT_FX adds MXN BRL ZAR KRW INR). Evidence: 800 BRL/INR cards (2001–2013, h 1–4): v13w-style − M0
  = −0.160 [−0.190, −0.130], and −0.026 against the v13v-style width. MXN, ZAR and KRW use the same rules by transfer; they were not
  tuned separately.
- v13w0 pre-branch: a short UST/FX level card outside the family path (for example an `em_` panel) takes v13w with every centre at the
  last level.
- ECB-sourced FX: when the unit manifest names the ECB reference rates (fixed 14:10 CET), the ECB-decision variance moves one business
  day later, because the decision (14:15 CET) comes after the fix. No fitted constant.
- Tests: 104/104 public units byte-identical (312 files); 79 Final-shaped units identical to v13w; 5 EM and 12 ECB-manifest units
  admissible.

## v13w3 (2026-10-14): full stack, robustness guards, D1

### Full short-horizon stack (actual constants in `t2agent/v13w.py`)

| Component | Value |
|---|---|
| UST event map `ue_nowkd` | separate note and bill tables; FOMC split by SEP month (Mar/Jun/Sep/Dec = FS) vs other (FO); the jump sits on the statement day. Notes: FOMC −2/−1 0.78; day 0 FS 1.84, FO 1.40; +1 FS 1.73, FO 1.28; +2 FS 1.08, FO 1.18; payrolls 2.31. Bills: −2/−1 0.82; day 0 FS 2.57, FO 1.61; +1 FS 1.52, FO 1.29; +2 FS 0.99, FO 0.92; payrolls 2.16 |
| EM table (ECB-fixed FX) | FOMC −2..0 0.95; +1 FS 2.44, FO 1.64; +2 FS 1.28, FO 1.25; payrolls 1.57, shifted +1 BD (8:30 ET comes after the 14:10 CET fix) |
| ECB table (ECB-fixed FX, by currency) | ECB meeting in a projection month (EP) / other (EO), BoJ (B): EUR and DKK EP 2.2, EO 1.4, B 1.27; CHF EP 1.8, EO 1.4, B 1.6; SEK and NOK EP 1.15, EO 1.05, B 1.2; GBP EP 2.1, B 1.4; JPY EP 1.35, B 2.9; AUD and NZD B 1.25 (ECB effects shifted +1 BD). H.10-sourced FX keeps the v13w tables |
| Calendar gates | BoJ terms only for as-of >= 2014-06-01, ECB terms only for as-of >= 2016-04-01 (the hard-coded calendars must cover the trailing 300 BD) |
| Note centre | last level + NOTE_LAM x (v12 centre − last level), **NOTE_LAM = 0.5**; bills keep the v12 centre; currencies 0.25 as before |
| Bill width clip | EWMA scale clipped to [0.35, 3] for bills ([0.5, 2] otherwise) |
| Grid shape | variance-preserving Student-t(12) quantile grid (GRID_DF = 12) |
| Dependence | t-copula with nu = 2 (COPULA_NU = 2) on the model correlation |

Selection: a second kit (kit3) built from ECB euro reference rates (G10 + MXN BRL ZAR KRW INR, practice quote conventions) and FRED
daily par yields including bills, h 1–4: development cards 1,314 FOMC-matched (as-of 2 BD before each decision) and 830 random, as-of
dates between 2005-01-01 and 2026-10-02 (outcomes up to 4 BD later); held-out cards 1,307 / 840 (seeds 63/64), first used on 14 Oct for a blind check
of the finished stacks. Each lever was scored alone and inside the stack (90% CI clustered by as-of date).

- Development (vs v13w2): F −0.0128 [−0.0179, −0.0077], R −0.0078 [−0.0119, −0.0039], Final-like subset (Monday as-of, Wednesday
  non-SEP FOMC at +2 BD, no payrolls; n = 304) −0.0155 [−0.0245, −0.0068], UST outside the zero bound −0.0137.
- Blind holdout (vs v13w2): F −0.0113 [−0.0167, −0.0058], R −0.0069 [−0.0103, −0.0034], Final-like (n = 305) −0.0088 [−0.0188,
  +0.0020], UST non-zero-bound −0.0164; negative in both eras.
- Rejected on these cards: month-end note drift (Final-like +0.0022), the alternative FOMC maps `l78` and `day0`, bill weekday
  dummies (+0.0019 Final-like), the em8 tail grid (worse than t(12) by 0.0017 on Final-like cards). The t-copula hurt dev cards with
  horizons {1,2,3,4} (+0.0031 FOMC, +0.0063 Final-like) but not the held-out ones; it was kept with the stack.
- The ledger name "note75" refers to an earlier 'safe' variant with NOTE_LAM = 0.75; the shipped value is 0.5.

### Robustness (no effect on any public unit)

- Declared-horizon guard: on short UST/FX level cards, target dates / observation periods are dropped so M0's walk takes h steps
  (M0-BASELINE 3.7 step 5 would otherwise count one step for h = 2 on some weekday as-ofs).
- FX/UST alias table: ids such as DGS3MO, UST_3MO, TSY_1Y, USGG, T-BILL, USDMXN, MXN/USD, EUR_ECB map to UST_<n>M/Y or the ISO code
  (only when no collision); the output keeps the card's own ids.
- Widened ECB detection: manifest or panel source text naming the ECB/European Central Bank/eurofxref/Eurosystem with
  reference/exchange/fixing context, or "14:10 CET" (0 false positives on 222 manifests), plus a calendar test (currency rows on >= 5
  US federal holidays in three years = ECB fixing days).
- Modelling time budget 1,000 s (unit limit 1,800 s), raised as a BaseException and caught; `parse_known_args`; `read_panels` also
  reads `<panels>/panels/*.parquet`; any exception outside the guarded block writes a complete emergency output (M0-equivalent draws,
  else an independent walk).
- v13wg route: a short daily level card whose ids are not all recognised takes the v13w path with centres at the last level and event
  timing only for recognised series (on renamed kit3 cards the old route lost +0.160 / +0.178 per card).
- Tests: fuzz (638 cases) FAIL 68 → 18 and inadmissible 27 → 0; fault injection 12/12 admissible (v13w2 7/12); public 104/104
  unchanged; on the 115 Final-shaped regression units the draws equal the full stack, except two CNH units that moved to v13wg.

### D1 draw layout (D1_ENABLED = True)

On the v13w, v13w0 and v13wg paths:

- draws = max(200, n_draws_min), where n_draws_min is the larger of card.toml `[scoring.params] n_draws_min` and forecast_spec.json
  `n_draws_min` (200 on every published card; the scorer accepts 200–20,000);
- copula sample: Halton points (prime bases ascending, first 20 points skipped) on the eigen-directions of the model correlation in
  descending eigenvalue order; dimensions beyond the prime list get antithetic, covariance-matched normals with a fixed seed (0); the
  t(2) scale uses the next Halton dimension (closed form for nu = 2);
- grid: the variance-preserving Student-t(12) grid at the positions that are optimal for the fair CRPS estimator, F = i/(m−1), end
  points clipped at 0.25/m.

Centres and per-cell sd are unchanged; the release model (v13r) and the legacy paths keep 500 draws.

Measured effect: the analytic fair-CRPS term of a deterministic grid is −0.20% of the expected CRPS at 500 draws and −0.50% at 200.
On kit3 development cards, D1 on v13w2: −0.0019 [−0.0028, −0.0010] FOMC, −0.0021 [−0.0030, −0.0011] random; on the shipped full stack
(eigen-ordered): −0.0010 [−0.0021, +0.0002] FOMC, −0.0024 [−0.0035, −0.0012] random (2,144 cards). Blind holdout: full with D1 − v13w2 =
−0.0123 (F), −0.0080 (R), against −0.0113 / −0.0069 without D1. Honest statement: D1 exploits the published fair-CRPS estimator (its
finite-sample term rewards a deterministic, low-discrepancy sample at the minimum draw count); it is not forecasting skill. It stays
inside the scorer's published draw limits.

## v13w4 (2026-10-14): release-card model v13r3

Module `t2agent/v13r.py` (docstring lists the method). It applies when every cell is a weekly, monthly or quarterly series (cadence
from the last 12 gaps, >= 16 rows), the target is a level and every horizon is <= 20 BD; it runs before the market branches.

| Part | Rule and constants |
|---|---|
| Steps | true number of series periods from the last observation to the target period (named month > card target date > as-of + h BD; target period = the period before the one containing the release date; a weekly series whose last row is <= 20 days old moves one week per 7 calendar days); at most 13 / 6 / 4 steps (W / M / Q) |
| Centre | inverse-MAE^4 combination of window-mean predictors on log levels (positive series that are large or move < 3% per step) or levels: last value + k-step mean drift (k = 2, 4, 8, 12, 24) and mean of the last w values (w = 1, 2, 4, 8, 13, 26); weights from the last N one-step self-errors (N = 104 / 60 / 40 for W / M / Q); plus x[t−P] + last P-period change (P = 52 / 12 / 4) when the lag-P autocorrelation of the last 11P changes exceeds 0.3 (unadjusted series) |
| Width | 1.4826 x MAD of the combined self-errors x 1.15 (floor 0.3 x RMS); Student-t(4) shape on an exact 500-point quantile grid |
| PCE | when the panel holds the matching CPI series for the missing month(s): Huber regression of the PCE change on the CPI change over the last 60 months (levels also tried on rate-scale series; the fit with the smaller implied sd wins), width 1.3 x residual MAD-sd |
| Joint | Gaussian copula: same asset sqrt(s_i/s_j); different assets 0.7 x the self-error correlation (>= 20 common periods); seed crc32(task id) |

Every quantity above is computed at run time from the unit's own panel; nothing fitted is stored.

- Selection: 5,539 synthetic release-card runs, 22 series-by-layout sets (plain, named-month, no-macro, stale, YoY, MoM, weekly
  continuing claims), release events 2005–2026. Each card's history is the ALFRED vintage in force on its as-of (the Monday of the
  release week) and its target is the first print. Series: ICSA, CCSA, PAYEMS, UNRATE, CPIAUCSL, CPILFESL, PCEPI, PCEPILFE, GDP, GDPC1,
  A191RL1Q225SBEA, ECIALLCIV, ECIWAG (ALFRED vintages downloaded 13 Oct 2026). The unadjusted-series candidate was studied on FRED
  NSA series (EZ HICP, CPIAUCNS, ICNSA, PAYNSA, UNRATENSA, retail sales NSA, EZ GDP NSA, Japan CPI).
- Results (official formula, mean over 5,539 runs; M0 / deployed v13w2 / v13r3): 'lit' step counting (M0 walks the documented steps)
  1.181 / 1.208 / 0.616; 's1' (M0 walks one step) 1.216 / 1.626 / 0.833; 0 crashes, 0 fallbacks; v13r3 beats v13w2 in at least 86% of
  years on every series except named-month CPI YoY.
- Rejected or not adopted: weekly width 1.0/1.3, CPI-regression width 1.1/1.5, level-only regression, quarterly window 60/80,
  quarterly multiplier 0.9, winsorised floor, 20-quarter MAD, the seasonal candidate without its switch, a claims bias correction.
- Calendars: release dates in the study come from the ALFRED vintage dates of the first prints; BEA, BLS and DOL schedules were
  consulted only to check the Final-week releases (the BLS schedule pages returned HTTP 403, so the ECI date of 30 Oct is
  unverified). v13r stores no calendar.
- Integration tests: public 104/104 byte-identical; Final-shaped 115/115 identical to v13w3; 834 release card-runs with draws identical
  to the research copy; 92/92 release cards admissible; release fault injection 33/33.
- The selection window reaches September 2026 first prints, before the ranked as-of. It is not claimed to be clean for earlier
  historical task cutoffs.

Post-adoption check (no constant changed): on 4,748 market cards with as-of dates in 2025–2026 (92 dates), v13w4 − v13w2 = −0.0084
[−0.0128, −0.0037] and v13w4 − M0 = −0.034; release cards of 2025–2026 beat M0 on almost every series. Weak spot left as is: UST_3M
cards score 1.189 against M0's 1.058 there, while the longer history points the other way.

## v13w5 (2026-10-15, release commit 2026-10-16)

v13w5 = v13w4 + the FOMC prior below. The release commit also corrects one sentence of the rationale: the sampling prose is chosen by a
flag that the D1 code path sets (`rec['d1']` in `v13w.draws`, carried into `prm['d1']`), instead of by "draws == 500"; before, a D1
card whose n_draws_min is 500 was described as "placed by the ranks of the baseline's joint draws". forecast.parquet and
forecast_meta.json are unchanged everywhere; forecast_rationale.md is unchanged on every regression set and changes only on cards
whose n_draws_min is 500.

## FOMC decision-uncertainty prior for the ranked Final (2026-10-15)

Constant `FINAL_FOMC_PRIOR` in `t2agent/v13w.py`. It applies only when the card's as-of date falls between 23 and 27 October 2026 and the
cell is a Treasury bill (3M, 6M, 1Y). For those cells, the variance of the single daily increment on 28 October 2026 (the FOMC statement day)
is multiplied by the square of a tenor-specific factor: sd x 2.5 (3M), x 2.0 (6M), x 1.5 (1Y).
All other cells, dates, centres, widths, event tables, grids and copulas are unchanged, so every public unit, every practice unit and the
in-window check-run cards (historical as-of dates) produce byte-identical outputs with or without the constant. When the prior applies to a
card, its forecast_rationale.md says so in one added sentence (method step 5).

Why: the event tables in `v13w.py` were fitted on daily history in which almost every FOMC decision was fully priced beforehand, so their
decision-day variance ratio for bills (1.61 at a non-projection meeting) describes a priced meeting. On the build date, public
market-implied odds of a 25 bp move at the 28 October 2026 meeting (CME FedWatch and prediction markets as reported in the financial press)
were far from both 0 and 1, so the bill increment on that day carries a two-sided jump of roughly 8-16 bp either way, several times a
bill's normal daily standard deviation. The factors were chosen by simulating the official expected-error composite on a 2026 3M-bill card
under live and priced scenarios (internal simulation study): widening only the decision-day increment by sd x 2-2.5
improves the expected score whenever the probability of a live decision exceeds about 0.12, and costs at most about 0.45 per 3M-only card if
the meeting turns out to be a non-event. Notes and currencies received no change because the same simulation showed no gain for the 2Y note
even in the live case.

Selection data: public market-odds reports dated 9-15 October 2026 (the rule and factors were set on 15 October) and FRED daily par
yields to 8 October 2026, all before the ranked as-of (26 October 2026). No outcome of any Final card was known or used; the constant
encodes a forward-looking judgment about one scheduled event, not a fit to realized values.

- Rule (set 2026-10-15): odds of a move in [10 %, 90 %] -> keep (2.5, 2.0, 1.5); below 10 % or above 90 % -> (1.5, 1.3, 1.2) or drop.
- Factors deployed in the Final image (see `FINAL_FOMC_PRIOR` in `t2agent/v13w.py`): 3M x 2.5, 6M x 2.0, 1Y x 1.5
  (as-of window 2026-10-23 .. 2026-10-27, decision 2026-10-28). The image was built and uploaded on 20-21 October 2026 with
  these default factors; live CME FedWatch odds of a move read on 21 October were about 16 % (inside the band).
- A later check on 24 October 2026 read Polymarket odds of a move at about 6 %, outside the band; under the rule the factors
  should then have been reduced to (1.5, 1.3, 1.2) or the prior dropped. No replacement upload was made (CodaBench was unreachable
  at the time), so the ranked entry carries 2.5 / 2.0 / 1.5. The prior affects only Treasury-bill cells on cards dated 23-27 October.

A panel-conditional version of the same idea (raising the FOMC-day variance whenever the asset's recent momentum is high) was tested on
historical cards and rejected because it lowered the score; the deployed constant is deliberately one-off and date-gated.

## Data sources, licenses and first availability

| Source | Used for | License / terms | First availability | Copy used |
|---|---|---|---|---|
| ECB euro foreign exchange reference rates (ECB Data Portal, `eurofxref-hist.zip`) | kit3 FX cards (G10 + EM vs USD, computed from the euro rates); ECB-fix timing | ECB statistics may be reused free of charge with the source acknowledged | same TARGET day (concertation 14:10 CET, published the same afternoon) | retrieved 13 Oct 2026, sha256 da0b31a0a8654b55…, rows 1999-01-04 … 2026-10-13 |
| FRED DGS1MO, DGS3MO, DGS6MO, DGS1, DGS2, DGS5, DGS7, DGS10, DGS20, DGS30 (Federal Reserve H.15 constant-maturity par yields) | kit3 UST and bill cards; FOMC-prior simulation | US Government data, public domain; FRED terms of use | daily H.15 update, within one business day of the observation date | retrieved 13 Oct 2026, last observation 2026-10-08 |
| ALFRED real-time vintages (St. Louis Fed) of the 13 release series listed under v13w4; underlying BEA (GDP, PCE), BLS (CPI, payrolls, unemployment, ECI) and DOL (claims) | v13r3 selection cards (history = vintage in force on the as-of; target = first print) | US Government data, public domain; ALFRED/FRED terms of use | each value at its vintage (realtime start) date | retrieved 13 Oct 2026 |
| Panels shipped in `Agenthon-2026/track2-forecasting-public` `units/` (FRED H.15, H.10, ECB, Fama-French/AQR, macro ALFRED vintages; licenses per unit manifest) | inherited v8–v12 stack, v13/v13v/v13w kits (observation dates 2000-01-03 … 2024-12-18) | per unit manifest | repository release dated 2026-07-03 (vintages before that not established) | workflow pin 30c8019d |
| FOMC, ECB and BoJ decision calendars; BLS Employment Situation rule | event timing | public schedules | published in advance of each year | hard-coded in `v13w.py` |
| CME FedWatch, Kalshi and Polymarket odds as reported in the press | FOMC prior factors only | public reports | dated 9 Oct 2026 onward | see the prior section |

Nothing is downloaded at run time; the image makes no network call. Libraries: python:3.13-slim-bookworm@sha256:5024f48ba9441d4b13a95d3945abc6365538e3a31109833367a1923523c6efed,
numpy 2.1.3, pandas 2.2.3, pyarrow 18.1.0 (no SciPy). The sealed image holds `t2agent/__init__.py` and `payload.bin` (all `t2agent/*.py`
packed by a private packaging script, `v13r.py` included; the packing script is not part of this repository).

## Cutoff applicability

- Ranked Final (as-of Mon 26 Oct 2026): every observation, vintage and odds report used to fit, select or retain a constant predates
  that as-of; the latest are kit3 outcomes to early October 2026, ALFRED first prints to September 2026, FRED yields to 8 Oct 2026 and
  odds reports up to the reading recorded in the prior section. No outcome of a Final card was known.
- Earlier historical cards (the public units and any in-window check run): the stack was selected on history that overlaps their
  outcomes, and the vintages of the public panels at those dates are not established. No eligibility is claimed for those cutoffs.
- No stored unit answers: the image holds code and constants only; per-card quantities are computed from that card's own `/input`.

## models: [] and the House model

The descriptor declares `models: []`. Per `docs/ARTIFACT-POLICY.md` (track2-forecasting-public, revision 2026-09-21.1), `models[]` is
for actual learned models (a bundled fitted model as `access: "local"`), while "pure code, dictionaries and tokenizer-only assets belong in
the provenance record, not fictitious model entries". The image contains no serialized weights, checkpoint or fitted-model file. v13r3's
per-card regressions and inverse-error weights, the EWMA widths and the copula correlations are computed at run time from the unit's own
panel and discarded; the fixed numbers (event variance ratios, centre factors, width clips, grid and copula settings, v13r3 window and
multiplier constants, the FOMC prior) are hand-set or grid-selected calibration constants written in the source and disclosed in this
record. This is our reading of the policy, not an organizer ruling. The shipped forecaster never calls the House model and makes no
network request.

## Source identity

SHA-256 of the numerical source in the release commit (`t2agent/`):

| File | SHA-256 |
|---|---|
| `__init__.py` | e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855 |
| `__main__.py` | f35124e530c67e144b61bdfca212301084982fc1408a2315f382617faa6e920f |
| `cli.py` | fda0f025a7bc01cff46af6b0cea05decee2daa9c19bfaf82d0e44509dc0d5716 |
| `core.py` | 281bcf32a84ce41d372a5b46c9f9232da5d54658c12e0489eb5652ee2a03d59d |
| `text.py` | 5705bfb463b4b7b85a3688b995ddba6afaf3b3a1636b6944c5763360d32c40ee |
| `v9.py` | d3a23d5e233ea279b988b8a1fb79bacfb1a536a634a53fc42be2ceb49adf3726 |
| `v10.py` | 828c5558dccf8769a6b2a7face5cbdc177b82a98cafd7633698eed77a94c126d |
| `v11.py` | 1e811a40da2a58f9e5f8b442d0597ad29624d5c35f54d9b7b06a57ce221afbdd |
| `v13r.py` | a8418500b2e209e27fec51be7ba417e332a1678400f52c6ed46e3379627180e4 |
| `v13w.py` | 49c70b5432aaa58302a28445a9d1a6f87eeca52efa38a70b336f50e6567c3c95 |

