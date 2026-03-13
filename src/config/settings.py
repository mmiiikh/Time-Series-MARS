from pathlib import Path
import os

# Корень проекта
BASE_DIR = Path(__file__).resolve().parents[2]

# Данные
DATA_PATH   = BASE_DIR / "data" / "raw" / "mars_data.xlsx"
RESULTS_DIR = BASE_DIR / "results"

# Модели
MODELS_DIR = BASE_DIR / "models"
SARIMA_DIR = MODELS_DIR / "sarima"
ML_DIR     = MODELS_DIR / "ml"
DL_DIR     = MODELS_DIR / "dl"

# MLflow
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
MLFLOW_EXPERIMENT   = os.getenv("MLFLOW_EXPERIMENT", "mars-forecasting")

# Колонки данных
TARGET_COL   = "MT_Volume KG"
DATE_COL     = "PER_SDESC"
CATEGORY_COL = "Row Labels"
CHANNEL_COL  = "MKT_SDESC"

EXOG_CANDIDATES = [
    "NT_Avg Line",
    "NT_Price per kg",
    "NT_Universe",
    "MT_Universe percent",
    "Frequency",
    "Penetration",
    "Spend per Trip",
    "Volume per Trip",
    "Total Mixed Chains - VoD (Vol)",
    "NT_CWD",
]

SEASONAL_PERIOD = 12
RANDOM_STATE    = 42

for _dir in [
    RESULTS_DIR / "part1",
    RESULTS_DIR / "part2",
    RESULTS_DIR / "part3",
    SARIMA_DIR,
    ML_DIR,
    DL_DIR,
]:
    _dir.mkdir(parents=True, exist_ok=True)