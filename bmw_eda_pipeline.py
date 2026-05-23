"""
BMW Used Cars — Exploratory Data Analysis & Feature Engineering
================================================================

Master's Degree project for Nuclio Digital School (Data Science).

Goal: prepare a real-world dataset of second-hand BMW vehicles for machine
learning. The pipeline covers:

    1. Initial exploration of the raw dataset.
    2. Duplicate detection.
    3. Null value handling — every decision is justified per variable
       depending on its distribution, the type of variable, the percentage
       of missing values and the available alternatives.
    4. Feature engineering (derived `antiguedad` column from registration
       and sale dates).
    5. Numerical and categorical variable analysis, including grouping of
       rare categories.
    6. Outlier detection on the target variable (`precio`).
    7. Target normalization with a logarithmic transformation.
    8. Bivariate visualizations (target vs. each feature).
    9. Encoding (boolean → int, one-hot for categoricals) and Min-Max
       scaling for numerical features.
    10. Export of the final, model-ready dataset.

Group project — 4 members.

NOTE: This script is the Python (.py) version of the original Google Colab
notebook. The Colab-specific code (Google Drive mounting) has been replaced
with local-friendly paths, and the dataframe variable chain
(bmw -> bmw_2 -> bmw3 -> ...) was unified into a single `df` for clarity.
"""

# =============================================================================
# IMPORTS
# =============================================================================
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler


# =============================================================================
# CONFIGURATION
# =============================================================================
INPUT_PATH = "data/bmw_pricing_v3.csv"
OUTPUT_PATH = "data/bmw_50filas.xlsx"
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)


# =============================================================================
# 1. DATA LOADING & INITIAL EXPLORATION
# =============================================================================
df = pd.read_csv(INPUT_PATH)

print("Shape:", df.shape)
print("\n--- df.info() ---")
df.info()
print("\n--- df.head() ---")
print(df.head())
print("\n--- df.describe() ---")
print(df.describe())


# =============================================================================
# 2. DUPLICATE DETECTION
# =============================================================================
print("\nDuplicate rows:", df.duplicated().sum())
print(df[df.duplicated(keep=False)])
# No duplicates found in the dataset.


# =============================================================================
# 3. NULL VALUE HANDLING
# =============================================================================
# Each variable is analyzed individually so the treatment of missing values
# is justified by the variable's nature, distribution and missing rate.
print("\nNulls per column:")
print(df.isnull().sum())


# --- 3.1 'marca' (brand) -----------------------------------------------------
# Since the dataset only contains BMW cars, this column has a single value
# and provides no predictive information. We drop it.
print("\n'marca' value counts:")
print(df['marca'].value_counts())
del df['marca']


# --- 3.2 'modelo' (model) ----------------------------------------------------
# The model is well distributed across many categories, so filling nulls
# with the mode would skew the data. We use a placeholder 'Sin Modelo'
# ("No model") so we keep these rows without forcing a specific category.
print("\n'modelo' value counts:")
print(df['modelo'].value_counts())
df['modelo'].fillna('Sin Modelo', inplace=True)


# --- 3.3 'km' (kilometers) ---------------------------------------------------
# Mean and median are similar, so the choice of imputation does not really
# matter. Negative kilometers are physically impossible, so we replace
# them (and the nulls) with the median.
df['km'].hist()
plt.title("Distribution of km")
plt.show()

print("\nkm mean:", df['km'].mean())
print("km median:", df['km'].median())
print("\nRows with negative km:")
print(df[df['km'] < 0])

df['km'].fillna(df['km'].median(), inplace=True)
df.loc[df['km'] < 0, 'km'] = df['km'].median()


# --- 3.4 'potencia' (power) --------------------------------------------------
# The distribution is non-normal, so replacing with the mean would be biased
# by extreme values. We use the median.
df['potencia'].hist()
plt.title("Distribution of potencia")
plt.show()
df['potencia'].fillna(df['potencia'].median(), inplace=True)


# --- 3.5 'tipo_gasolina' (fuel type) ----------------------------------------
# Mode is well differentiated, so we impute with it. While we are at it,
# we unify the inconsistent 'Diesel' / 'diesel' casing.
print("\n'tipo_gasolina' value counts:")
print(df['tipo_gasolina'].value_counts())

df['tipo_gasolina'].fillna('diesel', inplace=True)
df['tipo_gasolina'] = np.where(
    df['tipo_gasolina'] == 'Diesel', 'diesel', df['tipo_gasolina']
)


# --- 3.6 'color' -------------------------------------------------------------
# The mode is not clearly dominant and there are many nulls. We use a
# placeholder category instead of forcing one of the existing colors.
print("\n'color' value counts:")
print(df['color'].value_counts())
df['color'].fillna('sin_color', inplace=True)


# --- 3.7 'tipo_coche' (body type) -------------------------------------------
# Same rationale as 'color': no clear mode and many nulls → placeholder.
print("\n'tipo_coche' value counts:")
print(df['tipo_coche'].value_counts())
df['tipo_coche'].fillna('sin_tipo', inplace=True)


# --- 3.8 'fecha_registro' & 'fecha_venta' → 'antiguedad' --------------------
# Instead of working with two separate date columns, we engineer the
# 'antiguedad' (age in days) feature, which captures the meaningful
# relationship between the two dates. The original registration date is
# then dropped since it is not informative on its own.
df['fecha_registro'] = pd.to_datetime(df['fecha_registro'], errors='coerce')
df['fecha_venta'] = pd.to_datetime(df['fecha_venta'], errors='coerce')
df['antiguedad'] = (df['fecha_venta'] - df['fecha_registro']).dt.days
df.drop(columns=['fecha_registro'], inplace=True)


# --- 3.9 'antiguedad' --------------------------------------------------------
# Negative ages are impossible. We drop those rows, then fill remaining
# nulls with the median (the distribution is skewed).
print("\nRows with negative antiguedad:")
print(df[df['antiguedad'] < 0])
df = df.drop(df[df['antiguedad'] < 0].index)

df['antiguedad'].hist()
plt.title("Distribution of antiguedad")
plt.show()
df['antiguedad'].fillna(df['antiguedad'].median(), inplace=True)


# --- 3.10 'volante_regulable' (adjustable steering wheel, boolean) ----------
# Imputing a boolean variable with mean/mode does not make conceptual
# sense; since the number of nulls is small, we drop those rows.
print("\n'volante_regulable' value counts:")
print(df['volante_regulable'].value_counts())
df = df.dropna(subset=['volante_regulable'])


# --- 3.11 'aire_acondicionado' (air conditioning, boolean) ------------------
# Too many nulls to drop, and adding a new category would break the
# boolean nature of the variable. The mode is clearly dominant, so we
# impute with it.
print("\n'aire_acondicionado' value counts:")
print(df['aire_acondicionado'].value_counts())
df['aire_acondicionado'].fillna(True, inplace=True)


# --- 3.12 'camara_trasera' (rear camera, boolean) ---------------------------
# Only two possible values and a well-defined mode → impute with the mode.
print("\n'camara_trasera' value counts:")
print(df['camara_trasera'].value_counts())
df['camara_trasera'].fillna(False, inplace=True)


# --- 3.13 'asientos_traseros_plegables' (folding rear seats) ----------------
# More than 70% of the values are missing. Dropping the rows would
# discard most of the dataset, and imputing with the mode is questionable
# when nulls outnumber the mode itself. We drop the column.
null_pct = df['asientos_traseros_plegables'].isnull().sum() / df.shape[0] * 100
print(f"\n'asientos_traseros_plegables' null %: {null_pct:.1f}")
del df['asientos_traseros_plegables']


# --- 3.14 'elevalunas_electrico' (electric windows, boolean) ----------------
# Only two values and the mode is not strongly differentiated → drop nulls.
print("\n'elevalunas_electrico' value counts:")
print(df['elevalunas_electrico'].value_counts())
df = df.dropna(subset=['elevalunas_electrico'])


# --- 3.15 'bluetooth' --------------------------------------------------------
# Many nulls, but the mode is clearly dominant → impute with the mode.
print("\n'bluetooth' value counts:")
print(df['bluetooth'].value_counts())
df['bluetooth'].fillna(False, inplace=True)


# --- 3.16 'alerta_lim_velocidad' (speed limit alert, boolean) ---------------
# Too many nulls to drop. The mode is not strongly differentiated, but
# given that there is still meaningful data in the column, imputing with
# the mode is the most reasonable option.
df['alerta_lim_velocidad'].fillna(True, inplace=True)


# --- 3.17 'precio' (target) --------------------------------------------------
# Only a handful of nulls in the target. Since we never want to fabricate
# target values, we drop those rows instead.
df = df.dropna(subset=['precio'])


# --- 3.18 'fecha_venta' (sale date) -----------------------------------------
# Only a single null left → drop the row.
df = df.dropna(subset=['fecha_venta'])

print("\nNulls remaining after cleaning:")
print(df.isnull().sum())
df = df.reset_index(drop=True)


# =============================================================================
# 4. VARIABLE TYPE CLASSIFICATION
# =============================================================================
target = ['precio']


def obtener_lista_variables(dataset, target):
    """Return three lists splitting variables into numerical, boolean and
    categorical. Booleans are detected as numeric/bool variables with
    exactly two unique values."""
    lista_numericas = []
    lista_boolean = []
    lista_categoricas = []

    for col in dataset:
        kind = dataset[col].dtype.kind
        n_unique = len(dataset[col].unique())
        if (kind in ('f', 'i')) and n_unique != 2 and (col not in target):
            lista_numericas.append(col)
        elif (kind in ('f', 'i', 'b')) and n_unique == 2 and (col not in target):
            lista_boolean.append(col)
        elif (kind == 'O') and (col not in target):
            lista_categoricas.append(col)
    return lista_numericas, lista_boolean, lista_categoricas


lista_numericas, lista_boolean, lista_categoricas = obtener_lista_variables(df, target)
print("\nNumerical:", lista_numericas)
print("Boolean:", lista_boolean)
print("Categorical:", lista_categoricas)


# =============================================================================
# 5. NUMERICAL VARIABLE ANALYSIS
# =============================================================================
for col in lista_numericas:
    df.hist(col)
    plt.title(f"Distribution of {col}")
    plt.show()

print(df.describe())

# 'km' looks reasonable, no extreme values.
# 'fecha_venta' and 'antiguedad' do not show suspicious values.
# About 50% of values in 'antiguedad' are identical — this is a side effect
# of having imputed nulls with the median.

# --- 'potencia' anomaly: zero values ----------------------------------------
# A power of 0 is physically impossible. We drop those rows.
df = df.drop(df[df['potencia'] == 0].index)


# =============================================================================
# 6. CATEGORICAL VARIABLE ANALYSIS
# =============================================================================
# To avoid one-hot encoding generating hundreds of sparse columns, we
# group all values with very small counts into an 'otro' (other) category.
for col in lista_categoricas:
    print(df.value_counts(col))


# --- 6.1 'modelo' ------------------------------------------------------------
# Group every model with fewer than 100 occurrences into 'otro'.
df['modelo'] = df['modelo'].where(
    df['modelo'].isin(
        df['modelo'].value_counts()[df['modelo'].value_counts() >= 100].index
    ),
    'otro',
)


# --- 6.2 'tipo_gasolina' -----------------------------------------------------
# Drop the rare fuel types ('hybrid_petrol', 'electro') as they have too
# few samples to be meaningful for the model.
df = df.drop(df[df['tipo_gasolina'].isin(['hybrid_petrol', 'electro'])].index)


# --- 6.3 'color' -------------------------------------------------------------
# Group every color with fewer than 100 occurrences into 'otro'.
df['color'] = df['color'].where(
    df['color'].isin(
        df['color'].value_counts()[df['color'].value_counts() > 100].index
    ),
    'otro',
)


# --- 6.4 'tipo_coche' --------------------------------------------------------
# Same rationale, with a natural gap between counts below 100 and above 400.
df['tipo_coche'] = df['tipo_coche'].where(
    df['tipo_coche'].isin(
        df['tipo_coche'].value_counts()[df['tipo_coche'].value_counts() > 100].index
    ),
    'otro',
)


# =============================================================================
# 7. CORRELATION ANALYSIS (intermediate)
# =============================================================================
corr = df.corr(numeric_only=True)
print("\nCorrelation matrix:")
print(corr)
# Strong positive correlation between 'potencia' and 'precio' — power is
# a promising predictor of price.


# =============================================================================
# 8. TARGET ANALYSIS (precio)
# =============================================================================
# Detected during describe(): min price = 100, while the 25th percentile is
# 10,900 — clearly an outlier. The same happens with the max.
df['precio'].hist()
plt.title("Distribution of precio")
plt.show()

df['precio'].plot.box()
plt.title("Boxplot of precio")
plt.show()

print("\nLow-price outliers (<500):")
print(df[df['precio'] < 500])

# A second-hand car for less than 500 € is implausible, so we drop those
# rows. After analysis, prices above 100,000 € also turn out to be
# inconsistent with the rest of the vehicle's features (low power, high km,
# diesel) — we treat them as errors and drop them as well.
df = df.drop(df[df['precio'] < 500].index)

print("\nHigh-price outliers (>100,000):")
print(df[df['precio'] > 100000])
df = df.drop(df[df['precio'] > 100000].index)

# Price is heavily right-skewed, so we apply a logarithmic transformation
# to make it more symmetrical.
df['log_precio'] = np.log10(df['precio'])
df['log_precio'].hist()
plt.title("Distribution of log_precio")
plt.show()

sns.boxplot(x=df['log_precio'])
plt.title("Boxplot of log_precio")
plt.show()


# =============================================================================
# 9. TARGET vs FEATURES VISUALIZATIONS
# =============================================================================
# --- precio vs modelo --------------------------------------------------------
sns.violinplot(x="modelo", y="precio", data=df, palette="Wistia")
plt.title("precio vs modelo")
plt.show()
# Distributions are similar overall, but model '118' is the only one with
# multiple observations in the low-price range, and 'X5' has several
# observations above 30k.

# --- precio vs km ------------------------------------------------------------
sns.scatterplot(x="km", y="precio", data=df)
plt.title("precio vs km")
plt.show()
# Wide price range for cars with low km; as km increases, the maximum
# price drops.

# --- precio vs potencia ------------------------------------------------------
sns.scatterplot(x="potencia", y="precio", data=df)
plt.title("precio vs potencia")
plt.show()
# Higher power correlates with both higher minimum and maximum prices.

# --- precio vs tipo_gasolina -------------------------------------------------
sns.violinplot(x="tipo_gasolina", y="precio", data=df, palette="Wistia")
plt.title("precio vs tipo_gasolina")
plt.show()
# Similar distributions, with slightly more high-priced observations in
# 'petrol'.

# --- precio vs color ---------------------------------------------------------
sns.violinplot(x="color", y="log_precio", data=df, palette="Wistia")
plt.title("log_precio vs color")
plt.show()
# No meaningful difference across colors.

# --- precio vs tipo_coche ----------------------------------------------------
sns.violinplot(x="tipo_coche", y="precio", data=df, palette="Wistia")
plt.title("precio vs tipo_coche")
plt.show()
# 'estate' and 'hatchback' have noticeably different maximum prices; 'suv'
# has many high-priced observations.

# --- precio vs boolean features ---------------------------------------------
for boolean_col in ['volante_regulable', 'aire_acondicionado', 'camara_trasera',
                    'elevalunas_electrico', 'bluetooth', 'gps',
                    'alerta_lim_velocidad']:
    sns.violinplot(x=boolean_col, y="precio", data=df, palette="Wistia")
    plt.title(f"precio vs {boolean_col}")
    plt.show()
# Most boolean features show small but visible differences — the equipped
# version tends to skew towards slightly higher prices.

# --- precio vs fecha_venta ---------------------------------------------------
sns.scatterplot(x="fecha_venta", y="precio", data=df)
plt.title("precio vs fecha_venta")
plt.show()
# Almost all observations are from 2018, so the date itself is not
# informative — what matters is the engineered 'antiguedad' feature.

# --- precio vs antiguedad ----------------------------------------------------
sns.scatterplot(x="antiguedad", y="precio", data=df)
plt.title("precio vs antiguedad")
plt.show()
# Same pattern as km: higher age → lower maximum prices.


# =============================================================================
# 10. VARIABLE PROCESSING (encoding + scaling)
# =============================================================================
target = ['precio', 'log_precio']
lista_numericas, lista_boolean, lista_categoricas = obtener_lista_variables(df, target)

# --- 10.1 Boolean → int (easiest first) -------------------------------------
for col in lista_boolean:
    df[col] = df[col].astype(int)

# --- 10.2 One-hot encoding for categorical features -------------------------
df = pd.get_dummies(data=df, columns=lista_categoricas, dtype=int)

# --- 10.3 Date → numeric features -------------------------------------------
df['mes_venta'] = df['fecha_venta'].dt.month
df['dia_venta'] = df['fecha_venta'].dt.day
df['semana_venta'] = df['fecha_venta'].dt.weekday  # 0 = Monday, 6 = Sunday
df['año_venta'] = df['fecha_venta'].dt.year
df['trimestre_venta'] = df['fecha_venta'].dt.quarter

# 'dia_venta' has a single dominant value and adds nothing to the model.
# 'año_venta' is also very concentrated. We drop both, along with the
# original 'fecha_venta' column which we no longer need.
df = df.drop(columns=['dia_venta', 'fecha_venta', 'año_venta'])

# --- 10.4 Min-Max scaling for numerical features ----------------------------
scaler = MinMaxScaler()
df[lista_numericas] = scaler.fit_transform(df[lista_numericas])


# =============================================================================
# 11. FINAL CORRELATION ANALYSIS
# =============================================================================
corr2 = df.corr(numeric_only=True)
print("\nFinal correlation matrix shape:", corr2.shape)

# Keep only half of the matrix (lower triangle) to avoid duplicates
mask = np.triu(np.ones_like(corr2, dtype=bool))
corr_half = corr2.mask(mask)

# Convert to long-format table
corr_half = corr_half.stack().reset_index()
corr_half.columns = ['variable_1', 'variable_2', 'correlacion']

print("\nTop 10 positive correlations:")
print(corr_half.sort_values('correlacion', ascending=False).head(10))

print("\nTop 10 negative correlations:")
print(corr_half.sort_values('correlacion').head(10))

# Correlations of every variable against the target 'precio'
corr_precio = (
    df.corr()[['precio']]
    .drop('precio')
    .reset_index()
)
corr_precio.columns = ['variable', 'correlacion']
corr_precio['correlacion_abs'] = corr_precio['correlacion'].abs()

print("\nTop 10 features by absolute correlation with precio:")
print(corr_precio.sort_values('correlacion_abs', ascending=False).head(10))


# =============================================================================
# 12. EXPORT FINAL DATASET (first 51 rows as a sample)
# =============================================================================
sample = df[:51]
sample.to_excel(OUTPUT_PATH)
print(f"\nSample saved to {OUTPUT_PATH}")
print("Final dataset shape:", df.shape)
