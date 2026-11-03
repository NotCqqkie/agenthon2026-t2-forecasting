# Agenthon 2026 Track 2 (Forecasting): 3rd place entry

This is the source code of the forecaster that placed 3rd in Track 2 (Time-Series Forecasting) of
[Agenthon 2026](https://www.agenthon.net). It is released under the MIT licence as the winners' reproducibility package.

- Team: VerityLovityFalsityCruelty (Luke Finigan)
- Final entry: CodaBench submission 1006655, release v13w5
- Final image: `ghcr.io/notcqqkie/qfr-b@sha256:4195d2b76f9a86295481ad226a6e47d3e491fa1f1853784905b1752b6b77c667`

## What it does

For each forecast card the agent writes joint Monte Carlo draws for every target, plus `forecast_meta.json` and a short
`forecast_rationale.md`. Targets are Treasury yields, FX rates and official data releases. It starts from the organizers'
published text-blind baseline (M0, re-implemented from their documentation) and changes it only where tests showed a gain.

The Final cards were short-horizon cards (1 to 4 business days). On those, the main changes are:

- **Event-timed variance:** wider increments on FOMC, payrolls, ECB and BoJ decision days, from event tables fitted on daily history.
- **Short-horizon width:** an EWMA of recent volatility, a t-copula across cells, and an FX centre shrink.
- **D1 draw layout:** 200 draws placed on a deterministic quantile grid with a Halton copula. This uses a property of the official
  (fair) CRPS estimator, disclosed in METHOD.md.
- **Release model (v13r):** a first-release model for data-print cards (CPI, PCE, payrolls, unemployment, claims, GDP and others),
  selected on ALFRED vintages. These cards keep 500 draws.
- **FOMC prior:** a date-gated prior that widens only the 28 October 2026 decision-day increment of Treasury bills (3M, 6M, 1Y).
- **Fallback:** if the main method fails, the forecaster writes a complete baseline-equivalent output from the card's files.

Long-horizon family cards (the original public units) go through the inherited v9–v12 stack in `core.py`, `v9.py`, `v10.py`
and `v11.py`.

[METHOD.md](METHOD.md) is the full record written during development. It covers every step, every constant and the data used to
choose it, the holdout evidence, data sources with licences, and SHA-256 hashes of the files in `t2agent/`, which match this
repository. Some source comments refer to private development folders and files (`t2-work/…`, `claude2-lab`,
`ARTIFACT_PROVENANCE.md`). The source is published byte-identical to the Final entry, so those comments were left as they are.
METHOD.md is that provenance record.

The package has no learned model, weights or data files. The forecaster never calls a language model and makes no network
request. Everything it uses at run time comes from the card's own `/input` files. `t2agent/text.py` is an unused LLM client
that is never imported; see [NOTICE.md](NOTICE.md).

## Layout

| Path | What |
|---|---|
| `t2agent/` | the forecaster (Python, numpy/pandas/pyarrow only) |
| `Dockerfile` | builds the container the platform runs (`forecast` on PATH, qfbench2 interface 2.0 labels) |
| `scripts/run_unit.sh` | runs one unit under the platform's container settings |
| `.github/workflows/reproduce.yml` | builds this source, pulls the Final image, runs both on every public unit and checks that every output file is byte-identical |
| `METHOD.md` | method, constants, evidence, data sources and provenance |

## Reproduce

You need Docker (linux/amd64) and Python 3.11+ on the host.

```bash
docker build -t t2-source .
git clone https://github.com/Agenthon-2026/track2-forecasting-public
git -C track2-forecasting-public checkout 3b51e16b3f7c0fa0e56bcba323c2bb5e81b8bd56
scripts/run_unit.sh t2-source track2-forecasting-public/units/t2-F1-cad-boc-2017 out/t2-F1-cad-boc-2017
```

The output goes to `out/<unit>/` (`forecast.parquet`, `forecast_meta.json`, `forecast_rationale.md`). Inside the
container the command is the platform's:

```bash
forecast --panels /input/panels/ --text /input/text/ --asof <data_cutoff> --out /output/forecast.parquet
```

Seeds are fixed per card (`crc32` of the card id), so outputs are deterministic.

The `reproduce` workflow (run it from the Actions tab) runs this build and the published Final image on all 126 units at
the pinned commit and requires byte-identical output files. The pinned units are the 104 long-horizon public units plus
22 short-horizon practice cards, and the practice cards exercise the short-horizon and release-card paths above. The
ranked Final cards are the organizers' and are not public. The FOMC prior only fires on cards dated 23 to 27 October 2026,
so no public card exercises it. Its code is the `FINAL_FOMC_PRIOR` block in `t2agent/v13w.py`.

The Final image was built from the same `t2agent/` files. One extra packaging step stored the Python source in packed form
inside the image. That packaging script and its Dockerfile are not part of this repository. The packing does not change
behaviour, and the workflow above checks this.

## Data

This repository has no competition data. Units come from
[Agenthon-2026/track2-forecasting-public](https://github.com/Agenthon-2026/track2-forecasting-public), under that
repository's licence and each unit's manifest. Constants were selected on public data: FRED H.15 yields, ALFRED vintages
and ECB euro reference rates. Sources, terms and retrieval dates are in METHOD.md.

## Licence

MIT, see [LICENSE](LICENSE). Third-party notices are in [NOTICE.md](NOTICE.md).
