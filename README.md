# RecPlay

Recurrence plots (RPs) and recurrence quantification analysis (RQA) on financial and macroeconomic time series.
Part of the MaRBLe RQA research project (Tracks 2, 4 and 5: new RQA measures, RPs built with different
distance/divergence measures, and RPs as inputs to deep-learning models). The central question is whether
turning a time series into an RP improves outcome modelling over using the raw series.

## Research questions

- There is enough data in a single trading day to build RPs. Which combinations of data (price, volume,
  rolling volatility, indicators, rolling averages) give recurrence structure that carries useful information?
- Can an RP capture enough information to be used as input to a machine-learning model?
- Can entropy measures be used, and if so, how?
- Can RPs model outcomes probabilistically, rather than classifying whether the next move is up or down?

## Setup

Python 3.12. Third-party packages used by the notebooks:

```
pip install numpy pandas scipy matplotlib scikit-learn torch yfinance openpyxl jupyter
```

`torch` is only needed for `EucRPRepro` (expects a CUDA device). Versions are not pinned yet.

Run every notebook from its own folder (Jupyter and PyCharm do this by default). Paths to `data/` and
`results/` are relative to the notebook, e.g. `../../data/raw/fred`.

## Layout

```
RecPlay/
├── src/recplay/          shared RP / RQA code (imported by the 02_macro screening notebooks)
├── notebooks/
│   ├── 00_foundations/     exercises and first experiments
│   ├── 01_reproductions/   reproductions of published results
│   ├── 02_macro/           US macro series
│   ├── 03_equity_indices/  stock indices and RP computation benchmarks
│   └── 04_predictability/  RQA measures as predictability gauges
├── data/raw/             source data, not tracked in git
├── data/processed/       notebook caches, not tracked in git
└── results/tables/       tables written by notebooks
```

## Notebooks

| Folder | Notebook | What it does | Data |
|---|---|---|---|
| 00 | `ResearchBook` | Supervisor exercises: ε choice, noisy and random RPs, joint recurrence rates, the D₃ triple-dependence score | synthetic |
| 00 | `XOR_recurrence_demo` | Z = X ⊕ Y: dependence visible in the triple recurrence but not in any pair | synthetic |
| 00 | `recplay` | First distance matrix and thresholded RP on intraday prices; 3D recurrence onset surface over continuous ε (up to RR = 100 %) for the full series and a 20-point chunk, with timings | Yahoo, `CSG.AS` 2-min |
| 00 | `recplay_3d_viewer.py` (script) | Rotatable 3D onset surfaces for every sliding window of the day (default 20 samples, m = 2, τ = 1), slider / arrow keys to step through time, price panel shading the current window; `--help` for ticker, dates, window length, step, ε, fixed z scale | Yahoo, `CSG.AS` 2-min |
| 01 | `EucRPRepro` | Reproduction of Arratia & Sepúlveda: Euclidean RPs + CNN for the sign of the S&P 500 equity premium | Goyal–Welch 2025 |
| 02 | `CPI_UNRATE_FEDFUNDS_RR` | Inflation, unemployment and fed funds rate, 1954–2026: individual, pairwise and triple joint RPs; independence ratios, Theiler window, joint covariance/correlation; β-divergence RPs | FRED |
| 02 | `CPI_direction_RP_screening` | Which Euclidean sliding-window RP parameter (W, m, τ, target RR, input series) matters for predicting next month's YoY-inflation direction: RQA + RRc features, LightGBM with yearly expanding windows, variance decomposition and meta-model on 1975–1999, one final test on 2000–2026 | FRED |
| 02 | `CPI_direction_betaRP_screening` | Same screening with symmetric β-divergence RPs of exp(z) window states, β ∈ {−5, −2, −1, −0.5, 0.5, 1, 2, 5} as an extra factor | FRED |
| 02 | `CPI_direction_monthlyRP_horizons` | Direction of YoY inflation 1, 3, 6 and 12 months ahead: lag model with monthly inflation and the explicit base effect vs the same plus RP features of monthly inflation (window RQA, current-state RR, RRc, window mean/std, analogue forecasts); screening 1975–1999, one final test on 2000–2026 | FRED |
| 02 | `CPI_direction_betaRP_refinement` | β-divergence RPs of monthly inflation (W = 120, m = 2, τ = 1, rr = 0.05) for the same horizons: coarse-to-fine β search to depth 3 per horizon (8 β, then 8 β between the best and its better neighbour, twice) on 1975–1999, against Euclidean on z and on exp(z) and a β-profile model; every tried β scored once on 2000–2026 | FRED |
| 02 | `CPI_direction_embedding_sweep` | h = 6 and 12: ablation of the window mean/std vs RQA/RRc features; embedding (m, τ) chosen from 1954–1999 monthly inflation by AMI and false nearest neighbours; sweep of embedding span (1–24 months) for Euclidean on z and on exp(z), the best of 8 β and a β-profile model, each scored against lags+base+mean/std on 1975–1999 and 2000–2026 | FRED |
| 03 | `RUT_SP500_NDX_RR` | Monthly log-return recurrences of the Russell 2000, S&P 500 and Nasdaq-100 | Yahoo, `^RUT` `^GSPC` `^NDX` |
| 03 | `NDX_2025_5min_RP_benchmark` | Sliding-window RP compute time, Euclidean vs β-divergence (β = 1, 3, 7), calendar year 2025 | 5-min `^NDX` from a local CSV (not supplied) |
| 03 | `NDX_60d_5min_RP_benchmark` | Same benchmark on the last 60 days available from Yahoo | Yahoo, 5-min `^NDX` |
| 04 | `DETPredictability` | Whether DET combined with a second RQA measure tracks the actual predictability of chaotic and stochastic systems | synthetic (logistic map, Lorenz-63, …) |

## Library (`src/recplay`)

| Module | Contents |
|---|---|
| `embedding` | `embed`: time-delay embedding; `average_mutual_information`, `first_local_minimum` and `false_nearest_fraction` to choose τ and m |
| `recurrence` | distance matrices, symmetric β-divergence (any real β), ε set by a target recurrence rate (`threshold_by_rate`), single and joint recurrence matrices |
| `rqa` | diagonal/vertical line lengths, `rqa` measures, `k2_lines` |
| `connectivity` | `rr_critical`: RRc, the RR at which an 8-connected recurrence cluster first links the first δN rows to the last δN columns (union–find); `omega_ranks` for joint RPs |
| `surrogates` | `iaaft` surrogates |
| `windowed` | sliding-window RQA (plus the current state's recurrence rate, `RR_last`) and RRc (with IAAFT-surrogate null) feature frames; Euclidean on z or exp(z), or β-divergence distance |
| `analogue` | `analogue_frame`: the current state's recurrences with the whole past and the share of them followed by a rise h months later |
| `dependence` | independence ratios and joint covariance/correlation for pairs and triples |

Conventions: Euclidean distance unless stated; ε chosen per series so the recurrence rate equals a target
(default 0.10); rates counted over upper-triangle pairs (i < j); a Theiler window is applied by only counting
pairs with j − i ≥ `min_sep`.

## Data

`data/` is gitignored; recreate it as follows.

| Path | Source | Notes |
|---|---|---|
| `data/raw/fred/` | FRED CSV downloads: `CPIAUCSL`, `UNRATE`, `FEDFUNDS` (monthly) | Downloaded 18 Oct 2026. October 2025 CPI and UNRATE are blank: BLS did not collect them during the government shutdown. `CPI_UNRATE_FEDFUNDS_RR` interpolates and flags that month. |
| `data/raw/goyal_welch/Goyal_Welch_2025.xlsx` | Goyal & Welch equity-premium predictor data, 2025 update, `Monthly` sheet | 1871–2025 |
| `data/raw/yahoo/` | yfinance timezone cache | Created automatically; prices are downloaded live and change between runs |
| `data/processed/rp_screening/` | Caches of the screening notebooks: `iaaft/` window surrogates (shared), `euclidean/` and `beta/` RQA and RRc features | Delete a folder after changing the code that builds it |
| `data/processed/monthly_pi/` | Cache of `CPI_direction_monthlyRP_horizons`: `iaaft/` surrogates of monthly-inflation windows, `rqa/` and `rrc/` features | Delete after changing the code that builds it |
| `data/processed/beta_refinement/` | Cache of `CPI_direction_betaRP_refinement`: `iaaft/` surrogates, `rqa/` and `rrc/` features per arm (β or Euclidean) | Delete after changing the code that builds it |
| `data/processed/embedding_sweep/` | Cache of `CPI_direction_embedding_sweep`: `iaaft/` surrogates, `rqa/` and `rrc/` features per (m, τ, arm) | Delete after changing the code that builds it |

## Results

| File | Written by |
|---|---|
| `results/tables/det_predictability_results.csv` | `DETPredictability` |
| `results/tables/det_predictability_window_sweep.csv` | `DETPredictability` |

## References

- Marwan, N., Romano, M. C., Thiel, M., Kurths, J. (2007). Recurrence plots for the analysis of complex systems. *Physics Reports* 438, 237–329.
- Arratia, A., Sepúlveda, E. (2020). Convolutional neural networks, image recognition and financial time series forecasting. In *Mining Data for Financial Applications*, 60–69. DOI: 10.1007/978-3-030-37720-5_5.
- Theiler, J. (1986). Spurious dimension from correlation algorithms applied to limited time-series data. *Physical Review A* 34, 2427.
- Goyal, A., Welch, I. (2008). A comprehensive look at the empirical performance of equity premium prediction. *Review of Financial Studies* 21, 1455–1508. Data updated annually.
- Schreiber, T., Schmitz, A. (1996). Improved surrogate data for nonlinearity tests. *Physical Review Letters* 77, 635–638.
- Fraser, A. M., Swinney, H. L. (1986). Independent coordinates for strange attractors from mutual information. *Physical Review A* 33, 1134.
- Kennel, M. B., Brown, R., Abarbanel, H. D. I. (1992). Determining embedding dimension for phase-space reconstruction using a geometrical construction. *Physical Review A* 45, 3403.
- Federal Reserve Bank of St. Louis, FRED: https://fred.stlouisfed.org
