# BMW Used Cars — Exploratory Data Analysis & Feature Engineering

> **Master's Project** — Nuclio Digital School · Data Science module

This project takes a real-world dataset of second-hand **BMW** vehicles and prepares it for machine learning. The focus is on the **decisions** made at each step — how missing values are treated, how outliers are identified, how categorical variables are grouped, and how features are encoded and scaled. Every choice is justified empirically rather than applied mechanically.

---

## Project Overview

A used-car pricing dataset (~5,000 BMW vehicles) is processed through a complete data preparation pipeline. The goal is to turn raw, messy data into a clean, encoded and scaled dataset that any regression model could consume directly.

### Pipeline Steps

1. **Initial exploration** — shapes, types, descriptive statistics.
2. **Duplicate detection.**
3. **Null value handling** — every column is treated according to its nature, distribution and proportion of missing values:
   - Columns with a single dominant value or excessive nulls are **dropped**.
   - Numerical columns with skewed distributions are imputed with the **median**.
   - Boolean columns with a clear mode are imputed with the **mode**.
   - Categorical columns without a clear mode get a **placeholder category**.
   - The target variable (`precio`) is never imputed — rows with missing target are dropped instead.
4. **Feature engineering** — `antiguedad` (car age in days) is derived from the difference between registration and sale dates, replacing the raw date columns.
5. **Numerical variable analysis** — histograms and detection of physically impossible values (e.g. `potencia` = 0).
6. **Categorical variable grouping** — rare categories are merged into an `otro` bucket to avoid an explosion of one-hot columns.
7. **Outlier detection on the target** — prices below 500 € and above 100,000 € are identified as inconsistent with the rest of the vehicle's features and removed.
8. **Target normalization** — logarithmic transformation to correct the right-skewed price distribution.
9. **Bivariate exploration** — `precio` vs each feature using violinplots and scatterplots.
10. **Encoding & scaling** — booleans → int, categoricals → one-hot encoding, numericals → Min-Max scaling.
11. **Export** of a sample of the final, model-ready dataset.

---

## Key Decisions Made

The main value of this project is the **reasoning** behind each transformation. Some highlights:

| Column | Decision | Why |
|--------|----------|-----|
| `marca` | Dropped | All values are "BMW", no predictive signal. |
| `modelo` | Filled with `'Sin Modelo'` | Many categories, no clear mode → placeholder preserves the row without forcing a wrong category. |
| `km` | Negatives + nulls → median | Negative km are impossible; median is robust to outliers. |
| `potencia` | Nulls → median; rows with 0 dropped | Power of 0 is impossible; distribution is skewed. |
| `tipo_gasolina` | Nulls → mode; `'Diesel'` and `'diesel'` unified | Mixed casing was producing duplicate categories. |
| `asientos_traseros_plegables` | Column dropped | >70% missing — neither imputation nor row drop is reasonable. |
| `fecha_registro` + `fecha_venta` | Replaced by `antiguedad` | The relationship between the dates is what carries the predictive signal. |
| `precio` | Outliers <500 € and >100,000 € dropped | Inconsistent with the rest of the vehicle's features. |
| `precio` | Log transform | Right-skewed distribution → symmetrize via `log10`. |

---

## Technologies Used

- **Python 3.10+**
- **pandas** — data manipulation
- **NumPy** — numerical operations
- **Matplotlib** & **seaborn** — visualizations
- **scikit-learn** — `MinMaxScaler` for feature scaling
- **openpyxl** — Excel export

---

## Repository Structure

```
bmw-car-price-eda/
├── bmw_eda_pipeline.py        # Main script (full pipeline, end to end)
├── data/
│   └── bmw_50filas.xlsx       # Sample output — first 51 rows of the final dataset
├── requirements.txt           # Python dependencies
├── README.md                  # This file
└── .gitignore                 # Files excluded from version control
```

---

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/bmw-car-price-eda.git
cd bmw-car-price-eda
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate    # On Windows: venv\Scripts\activate
```

### 3. Install the dependencies

```bash
pip install -r requirements.txt
```

### 4. Provide the input dataset

The script expects the raw BMW pricing dataset at `data/bmw_pricing_v3.csv`. This dataset was provided by the master's program and is **not included in this repository**. To run the pipeline yourself, place a compatible CSV at that path with the following columns:

`marca`, `modelo`, `km`, `potencia`, `tipo_gasolina`, `color`, `tipo_coche`, `fecha_registro`, `fecha_venta`, `precio`, plus the boolean equipment columns (`volante_regulable`, `aire_acondicionado`, `camara_trasera`, `asientos_traseros_plegables`, `elevalunas_electrico`, `bluetooth`, `gps`, `alerta_lim_velocidad`).

### 5. Run the script

```bash
python bmw_eda_pipeline.py
```

The script prints every diagnostic step to the console and displays a series of plots. The first 51 rows of the final, model-ready dataset are saved to `data/bmw_50filas.xlsx` (see [data/bmw_50filas.xlsx](data/bmw_50filas.xlsx) for an example output).

---

## Sample Output

The included `data/bmw_50filas.xlsx` shows the result of the pipeline: a fully numerical, scaled and encoded dataset where every column is ready to be fed into a regression model. Notable transformations visible in the sample:

- Numerical features (`km`, `potencia`, `antiguedad`) are scaled to the [0, 1] range.
- Categorical features (`modelo`, `tipo_gasolina`, `color`, `tipo_coche`) are expanded into binary columns.
- Booleans (`aire_acondicionado`, `camara_trasera`, etc.) are encoded as 0/1.
- The original `precio` is preserved alongside its log-transformed version `log_precio`.

---

## License

This project was developed for educational purposes within the Master's program at Nuclio Digital School.
