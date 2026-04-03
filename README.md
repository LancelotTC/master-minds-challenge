# DATA VISUALISATION

Exploratory Data Analysis on flight movement data from **Lyon Saint-Exupéry Airport** (`database.db`, table `mouvements_aero_insa`).  
The dataset contains **364,623 rows × 195 columns** covering scheduled/actual times, passenger counts, airlines, aircraft types, routes, terminals, and enriched external features (weather, public holidays, school holidays).

---

## 📁 Folder Structure

```
master-minds-challenge/
│
├── database.db                   # SQLite database (gitignored)
├── airport-flight-dataset.csv    # Raw CSV export of the dataset
│
├── data_quality.ipynb            # 1 — Missing values, duplicates, dtype validation
├── descriptive_stats.ipynb       # 2 — Shape overview, numeric & categorical stats
├── data_visualization.ipynb      # 3 — Visual exploration (13 charts)
├── correlation.ipynb             # 4 — Feature correlation & ranking vs NbPaxTotal
├── outlier_analysis.ipynb        # 5 — Outlier detection on numeric columns
│
├── update_datasets.py            # Script to rebuild database from BigQuery + external APIs
├── utils.py                      # Shared utilities (SqliteManager, ProgressBar)
├── requirements.txt              # Python dependencies
│
└── outputs/
    ├── data_quality/
    │   ├── missing_report.csv            # Per-column null + string-placeholder counts
    │   ├── 01_missing_values.png         # Stacked bar: null vs string placeholders
    │   └── 02_completeness.png           # Completeness % for all 195 columns
    │
    ├── descriptive_stats/
    │   ├── column_overview.csv           # dtype, null count, unique count (195 cols)
    │   ├── numeric_stats.csv             # describe + skewness + kurtosis
    │   ├── categorical_value_counts.csv  # Top-10 values per categorical column
    │   └── nbpaxtotal_stats.csv          # Detailed stats for NbPaxTotal
    │
    ├── correlation/
    │   ├── feature_ranking_nbpax.csv     # Features ranked by correlation with NbPaxTotal
    │   ├── 01_mean_nbpax_by_category.png
    │   ├── 02_correlation_ranking.png
    │   ├── 03_heatmap.png
    │   ├── 04_scatter_plots.png
    │   └── 05_box_plots.png
    │
    ├── outlier_analysis/
    │   ├── 01_boxplots.png
    │   └── 02_nbpax_distribution.png
    │
    └── visualization/
        ├── 01_arrivals_vs_departures.png
        ├── 02_top_airlines.png
        ├── 03_flights_per_month.png
        ├── 04_flights_by_hour.png
        ├── 05_by_terminal.png
        ├── 06_top_origin_airports.png
        ├── 07_delay_analysis.png
        ├── 08_nbpax_distribution.png
        ├── 09_nbpax_temporal.png
        ├── 10_nbpax_by_airline.png
        ├── 11_nbpax_by_aircraft.png
        ├── 12_nbpax_terminal_trafic.png
        └── 13_nbpax_top_routes.png
```

---

## How to Use

### 1. Prerequisites

```bash
pip install -r requirements.txt
```

> Requires **Python 3.10+** and a `database.db` SQLite file in the project root.  
> The DB is gitignored — obtain it separately or regenerate it with `update_datasets.py`.

### 2. (Optional) Rebuild the database

`update_datasets.py` pulls raw flight data from **Google BigQuery**, enriches it with weather, French public holidays and school holidays, then writes `database.db`.

```bash
# Set credentials first
cp .env.example .env   # fill in GOOGLE_APPLICATION_CREDENTIALS etc.

python update_datasets.py
```

### 3. Run the notebooks (recommended order)

Open any notebook in VS Code / JupyterLab and **Run All Cells**.  
Each notebook is self-contained — it connects to `database.db`, runs its analysis, and writes outputs to its dedicated folder under `outputs/`.

| Step | Notebook                   | What it does                                                                                                                                         | Outputs                      |
| ---- | -------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------- |
| 1    | `data_quality.ipynb`       | Extended missing-value detection (null + string placeholders like `""`, `"NaN"`, `"N/A"`, `"-"`), duplicates, dtype validation, invalid value checks | `outputs/data_quality/`      |
| 2    | `descriptive_stats.ipynb`  | Column overview, numeric describe + skewness/kurtosis, categorical value counts, NbPaxTotal deep-dive                                                | `outputs/descriptive_stats/` |
| 3    | `data_visualization.ipynb` | 13 charts: traffic by direction/airline/hour/terminal/route, delay analysis, passenger distributions                                                 | `outputs/visualization/`     |
| 4    | `correlation.ipynb`        | Pearson/Spearman correlations and categorical ANOVA for all 195 features vs `NbPaxTotal`, ranked feature importance                                  | `outputs/correlation/`       |
| 5    | `outlier_analysis.ipynb`   | IQR-based outlier detection on numeric columns, distribution plots                                                                                   | `outputs/outlier_analysis/`  |

---

## 📊 Dataset Summary

| Property        | Value                                                                                                                                  |
| --------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| Source          | Lyon Saint-Exupéry Airport (INSA challenge)                                                                                            |
| Table           | `mouvements_aero_insa`                                                                                                                 |
| Rows            | 364,623                                                                                                                                |
| Columns         | 195                                                                                                                                    |
| Target variable | `NbPaxTotal` (total passengers per flight)                                                                                             |
| Time coverage   | See `LTScheduledDatetime` range in `data_quality.ipynb`                                                                                |
| Key features    | Direction (A/D), airline, aircraft type, terminal, origin/destination airport, scheduled & actual times, delay, weather, holiday flags |

> ⚠️ **Missing data note:** Many columns contain string-encoded missing values (`""`, `"NaN"`, `"N/A"`, `"unknown"`, etc.) in addition to true `null`.  
> The `data_quality.ipynb` notebook handles both forms — see `outputs/data_quality/missing_report.csv` for the full breakdown.
