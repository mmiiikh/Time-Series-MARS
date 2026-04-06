from src.data.load_data import load_data, create_series_dict
from src.config.settings import DATA_PATH
from src.forecasting.sarima import (
    train_arima, train_seasonal_naive, compare_all_models, load_manifest
)
import pickle, json
from pathlib import Path

# Загружаем данные
df          = load_data(str(DATA_PATH))
series_dict = create_series_dict(df)

# Загружаем уже обученные результаты из манифеста
# (вместо повторного обучения SARIMA/SARIMAX)
manifest = load_manifest()

# Обучаем только ARIMA
results_arima = train_arima(series_dict, test_size=12)
results_naive = train_seasonal_naive(series_dict, test_size=12)

# Для SARIMA/SARIMAX берём метрики из манифеста
results_sarima  = {}
results_sarimax = {}
for key_str, info in manifest.items():
    cat, ch = info["category"], info["channel"]
    key = (cat, ch)
    mape = info.get("mape")
    fake = {"metrics": {"mape": mape}, "model_type": info["best_model"]}
    if info["best_model"] == "SARIMAX":
        results_sarimax[key] = fake
    else:
        results_sarima[key] = fake

compare_all_models(results_sarimax, results_sarima, results_naive, results_arima)