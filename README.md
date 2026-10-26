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
├── src/recplay/          shared RP / RQA code (not yet imported by the notebooks)
├── notebooks/
│   ├── 00_foundations/     exercises and first experiments
│   ├── 01_reproductions/   reproductions of published results
│   ├── 02_macro/           US macro series
│   ├── 03_equity_indices/  stock indices and RP computation benchmarks
│   └── 04_predictability/  RQA measures as predictability gauges
├── data/raw/             source data, not tracked in git
└── results/tables/       tables written by notebooks
```

## Notebooks

| Folder | Notebook | What it does | Data |
|---|---|---|---|
| 00 | `ResearchBook` | Supervisor exercises: ε choice, noisy and random RPs, joint recurrence rates, the D₃ triple-dependence score | synthetic |
| 00 | `XOR_recurrence_demo` | Z = X ⊕ Y: dependence visible in the triple recurrence but not in any pair | synthetic |
| 00 | `recplay` | First distance matrix and thresholded RP on intraday prices | Yahoo, `CSG.AS` 2-min |
| 01 | `EucRPRepro` | Reproduction of Arratia & Sepúlveda: Euclidean RPs + CNN for the sign of the S&P 500 equity premium | Goyal–Welch 2025 |
| 02 | `CPI_UNRATE_FEDFUNDS_RR` | Inflation, unemployment and fed funds rate, 1954–2026: individual, pairwise and triple joint RPs; independence ratios, Theiler window, joint covariance/correlation; β-divergence RPs | FRED |
| 03 | `RUT_SP500_NDX_RR` | Monthly log-return recurrences of the Russell 2000, S&P 500 and Nasdaq-100 | Yahoo, `^RUT` `^GSPC` `^NDX` |
| 03 | `NDX_2025_5min_RP_benchmark` | Sliding-window RP compute time, Euclidean vs β-divergence (β = 1, 3, 7), calendar year 2025 | 5-min `^NDX` from a local CSV (not supplied) |
| 03 | `NDX_60d_5min_RP_benchmark` | Same benchmark on the last 60 days available from Yahoo | Yahoo, 5-min `^NDX` |
| 04 | `DETPredictability` | Whether DET combined with a second RQA measure tracks the actual predictability of chaotic and stochastic systems | synthetic (logistic map, Lorenz-63, …) |

## Library (`src/recplay`)

| Module | Contents |
|---|---|
| `embedding` | `embed`: time-delay embedding |
| `recurrence` | distance matrices, ε set by a target recurrence rate, single and joint recurrence matrices |
| `rqa` | diagonal/vertical line lengths, `rqa` measures, `k2_lines` |
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
- Federal Reserve Bank of St. Louis, FRED: https://fred.stlouisfed.org
