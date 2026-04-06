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
RANDOM_STATE = 42

for _dir in [
    RESULTS_DIR / "part1",
    RESULTS_DIR / "part2",
    RESULTS_DIR / "part3",
    SARIMA_DIR,
    ML_DIR,
    DL_DIR,
]:
    _dir.mkdir(parents=True, exist_ok=True)

# Файл данных для ML части — отдельный файл с каналами
ML_DATA_FILE = BASE_DIR / "data" / "raw" / "Mars_2_Data_Chanels.xlsx"

# Пути для ML артефактов — алиасы для удобства в ML скриптах
ML_MODELS_DIR = ML_DIR
ML_RESULTS_DIR = RESULTS_DIR / "part2"

# MLflow эксперименты для ML части
MLFLOW_EXPERIMENT_ML = "mars_ml_models"
MLFLOW_EXPERIMENT_TUNING = "mars_ml_tuning"
MLFLOW_EXPERIMENT_EXP = "mars_ml_experiments"

# Экзогенные переменные для ML — без "Total Mixed Chains - VoD (Vol)" (45% пропусков)
EXOG_COLS = [
    "NT_CWD", "NT_Avg Line", "NT_Price per kg", "NT_Universe",
    "MT_Universe percent", "Frequency", "Penetration",
    "Spend per Trip", "Volume per Trip",
]

# Параметры обучения ML моделей
TEST_SIZE = 12  # последние 12 месяцев = тестовая выборка
N_FOLDS = 3  # walk-forward CV фолды
HORIZON = 12  # горизонт прогноза на будущее

# Горизонты для CV оценки
# h=12 нельзя добавить: при test_size=12 shift(-12) даёт NaN для всех строк
CV_HORIZONS = [1, 3, 6]

# COVID период
COVID_START = "2020-03-01"
COVID_END = "2020-06-01"

# Гиперпараметры по умолчанию
LGBM_DEFAULT_PARAMS = dict(
    n_estimators=500, learning_rate=0.03, max_depth=5, num_leaves=31,
    min_child_samples=5, subsample=0.8, colsample_bytree=0.8,
    reg_lambda=1.0, random_state=RANDOM_STATE, verbose=-1,
)

XGB_DEFAULT_PARAMS = dict(
    n_estimators=500, learning_rate=0.03, max_depth=4, subsample=0.8,
    colsample_bytree=0.8, reg_lambda=1.0,
    random_state=RANDOM_STATE, verbosity=0,
)

EN_DEFAULT_PARAMS = dict(
    l1_ratio=[0.1, 0.5, 0.7, 0.9, 1.0],
    cv=3, max_iter=5000, random_state=RANDOM_STATE,
)

# Optuna — количество триалов
OPTUNA_N_TRIALS_LGBM = 50
OPTUNA_N_TRIALS_XGB = 40
OPTUNA_N_TRIALS_EN = 30