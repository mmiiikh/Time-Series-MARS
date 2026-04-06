"""
src/forecasting/ml_v2_forecast.py

Инференс ML v2 модели (DirectForecasterV2) для API.

Загружает финальную модель из ml_v2_{config}_metadata.json,
читает pkl победителя и генерирует прогноз.

Используется из routes.py при model_class="ml_v2".
"""

from __future__ import annotations

import json
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from src.config.settings import ML_MODELS_DIR, ML_RESULTS_DIR, TARGET_COL, HORIZON

warnings.filterwarnings("ignore")

# Конфигурация которую используем в продакшне
# Выбирается по результатам mars_ml_exog_v2_comparison
PROD_CONFIG = "vif_exog"


def _load_metadata(config: str = PROD_CONFIG) -> dict:
    """Загружает метаданные финальной ML v2 модели."""
    path = ML_MODELS_DIR / f"ml_v2_{config}_metadata.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Метаданные ML v2 не найдены: {path}. "
            f"Запустите src/training/train_ml_exog_v2.py"
        )
    with open(path) as f:
        return json.load(f)


def _load_model(config: str = PROD_CONFIG):
    """Загружает pkl финальной full-модели."""
    meta = _load_metadata(config)
    pkl_path = Path(meta["winner_pkl_full"])
    if not pkl_path.exists():
        raise FileNotFoundError(f"pkl модели не найден: {pkl_path}")
    with open(pkl_path, "rb") as f:
        return pickle.load(f), meta


def _get_mlflow_mape(category: str, channel: str) -> dict:
    """
    Читает MAPE из MLflow mars_ml_exog_v2_comparison для данного ряда.
    Возвращает {no_exog, partial_exog, full_exog, config_name} или {}.
    """
    try:
        import mlflow
        from src.config.settings import MLFLOW_TRACKING_URI
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = mlflow.tracking.MlflowClient()
        exp    = client.get_experiment_by_name("mars_ml_exog_v2_comparison")
        if exp is None:
            return {}
        runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            filter_string=f"tags.category = '{category}' and tags.channel = '{channel}'",
            max_results=10,
        )
        # Ищем прежде всего vif_exog, потом all_exog
        for config_pref in [PROD_CONFIG, "all_exog"]:
            for run in runs:
                if run.data.tags.get("config_name") == config_pref:
                    m = run.data.metrics
                    no_exog = m.get("ml_v2_mape_no_exog")
                    partial = m.get("ml_v2_mape_partial_exog")
                    full    = m.get("ml_v2_mape_full_exog")
                    return {
                        "no_exog":      round(no_exog, 2) if no_exog else None,
                        "partial_exog": round(partial, 2) if partial else None,
                        "full_exog":    round(full, 2)    if full    else None,
                        "config_name":  config_pref,
                        "winner_model": run.data.tags.get("winner_v2_model", ""),
                        "architecture": run.data.tags.get("architecture", ""),
                        "selected_exog": run.data.tags.get("selected_exog", ""),
                    }
    except Exception:
        pass
    # Fallback: читаем из CSV если MLflow недоступен
    try:
        csv_path = ML_RESULTS_DIR / f"test_metrics_v2_{PROD_CONFIG}.csv"
        if not csv_path.exists():
            csv_path = ML_RESULTS_DIR / "test_metrics_v2.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            sub = df[(df["category"] == category) & (df["channel"] == channel)]
            result = {}
            for mode in ["no_exog", "partial_exog", "full_exog"]:
                row = sub[sub["mode"] == mode]
                if len(row) > 0:
                    result[mode] = round(float(row["mape"].values[0]), 2)
            if result:
                result["config_name"] = PROD_CONFIG
                return result
    except Exception:
        pass
    return {}


def get_sarima_mape(category: str, channel: str) -> float | None:
    """Читает MAPE SARIMA из MLflow mars-forecasting."""
    try:
        import mlflow
        from src.config.settings import MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = mlflow.tracking.MlflowClient()
        for exp_name in [MLFLOW_EXPERIMENT, "mars-forecasting", "mars_sarima"]:
            exp = client.get_experiment_by_name(exp_name)
            if exp is None:
                continue
            runs = client.search_runs(
                experiment_ids=[exp.experiment_id], max_results=2000
            )
            for run in runs:
                parts = run.data.tags.get("mlflow.runName", "").split("|")
                if len(parts) >= 3:
                    if parts[1].strip() == category and parts[2].strip() == channel:
                        mape = run.data.metrics.get("best_mape")
                        if mape is not None:
                            return round(float(mape), 2)
    except Exception:
        pass
    return None


def get_comparison(category: str, channel: str) -> dict:
    """
    Возвращает сравнение MAPE всех доступных моделей для ряда.
    Используется для рекомендации лучшей модели.

    Returns:
        {
          "sarima_mape":         float | None,
          "ml_v2_no_exog":       float | None,
          "ml_v2_partial_exog":  float | None,
          "ml_v2_full_exog":     float | None,
          "best_model":          "sarima" | "ml_v2",
          "best_mape":           float | None,
          "recommendation":      str,
          "selected_exog":       list[str],
        }
    """
    sarima_mape = get_sarima_mape(category, channel)
    ml_info     = _get_mlflow_mape(category, channel)
    ml_no_exog  = ml_info.get("no_exog")

    # Определяем лучшую модель по no_exog сценарию (честное сравнение)
    best_model = None
    best_mape  = None

    candidates = {}
    if sarima_mape is not None:
        candidates["sarima"] = sarima_mape
    if ml_no_exog is not None:
        candidates["ml_v2"] = ml_no_exog

    if candidates:
        best_model = min(candidates, key=candidates.get)
        best_mape  = candidates[best_model]

    # Формируем текстовую рекомендацию
    if best_model == "sarima":
        model_label = "SARIMA/SARIMAX"
        recommendation = (
            f"Для этого ряда рекомендуется {model_label} "
            f"(MAPE={sarima_mape:.1f}% на тесте)"
        )
    elif best_model == "ml_v2":
        recommendation = (
            f"Для этого ряда рекомендуется ML v2 "
            f"(MAPE={ml_no_exog:.1f}% без экзогенных). "
            f"С плановыми данными MAPE может быть ниже."
        )
    else:
        recommendation = "Нет данных о качестве моделей."

    # Список экзогенных из метаданных
    selected_exog: list[str] = []
    try:
        meta = _load_metadata(PROD_CONFIG)
        selected_exog = meta.get("selected_exog", [])
    except Exception:
        pass

    return {
        "sarima_mape":        sarima_mape,
        "ml_v2_no_exog":      ml_no_exog,
        "ml_v2_partial_exog": ml_info.get("partial_exog"),
        "ml_v2_full_exog":    ml_info.get("full_exog"),
        "best_model":         best_model,
        "best_mape":          best_mape,
        "recommendation":     recommendation,
        "selected_exog":      selected_exog,
        "config_name":        ml_info.get("config_name", PROD_CONFIG),
        "winner_v2_model":    ml_info.get("winner_model", ""),
    }


def forecast_ml_v2(
    series_dict: dict,
    category: str,
    channel: str,
    horizon: int,
    user_exog: dict[str, list[float]] | None = None,
    config: str = PROD_CONFIG,
) -> dict:
    """
    Генерирует прогноз ML v2 модели.

    Args:
        series_dict:  сырые данные {(cat, ch): DataFrame}
        category:     категория ряда
        channel:      канал ряда
        horizon:      горизонт прогноза (месяцев)
        user_exog:    плановые значения от пользователя
                      {col_name: [v1, v2, ..., v_horizon]}
                      Например: {"NT_Price per kg": [100, 102, 103, ...]}
        config:       имя конфигурации ("vif_exog" или "all_exog")

    Returns:
        dict с полями forecast, model_type, model_spec, test_mape, exog_cols
    """
    fc_model, meta = _load_model(config)

    key = (category, channel)
    if key not in series_dict:
        raise ValueError(f"Ряд '{category}|{channel}' не найден")

    series_df      = series_dict[key]
    base_features  = meta["base_features"]
    selected_exog  = meta.get("selected_exog", [])
    last_value_per = {
        tuple(k.split("|")): v
        for k, v in meta.get("last_value_per_series", {}).items()
    }
    global_last    = meta.get("global_last", {})

    # Восстанавливаем last_value_per_series в FC модели
    fc_model.last_value_per_series = last_value_per
    fc_model.global_last           = global_last

    # Строим x_T — признаки последней известной точки
    from src.data.load_data import load_data, create_series_dict
    from src.data.preprocess import build_global_dataset

    lag_ranges = meta.get("ccf_lag_ranges", {})
    df_raw     = load_data(str(__import__("src.config.settings", fromlist=["ML_DATA_FILE"]).ML_DATA_FILE))
    sd         = create_series_dict(df_raw)
    global_df, _, _, _ = build_global_dataset(sd, lag_ranges)

    # Берём строку с нужным рядом
    series_rows = global_df[
        (global_df["_category"] == category) &
        (global_df["_channel"]  == channel)
    ]
    if series_rows.empty:
        raise ValueError(f"Нет данных в global_df для '{category}|{channel}'")

    # x_T — последняя строка доступных данных
    avail_feats = [f for f in base_features if f in series_rows.columns]
    x_T = series_rows[avail_feats].fillna(0).iloc[[-1]].copy()

    last_date = pd.Timestamp(series_rows["_date"].max())

    # Готовим user_exog в формат {col: [v1..v_horizon]}
    prepared_user: dict[str, list[float]] = {}
    if user_exog:
        for col, vals in user_exog.items():
            if col in selected_exog:
                vals_full = list(vals)
                if len(vals_full) < horizon:
                    last_v = vals_full[-1] if vals_full else 0.0
                    vals_full = vals_full + [last_v] * (horizon - len(vals_full))
                prepared_user[col] = [float(v) for v in vals_full[:horizon]]

    forecast_points = []
    for h in range(1, horizon + 1):
        fd = last_date + pd.DateOffset(months=h)

        x_h = x_T.copy()
        for feat, val in [
            ("month",       int(fd.month)),
            ("month_sin",   float(np.sin(2 * np.pi * fd.month / 12))),
            ("month_cos",   float(np.cos(2 * np.pi * fd.month / 12))),
            ("quarter",     int(fd.quarter)),
            ("quarter_sin", float(np.sin(2 * np.pi * fd.quarter / 4))),
            ("quarter_cos", float(np.cos(2 * np.pi * fd.quarter / 4))),
            ("is_q4",       int(fd.month >= 10)),
            ("is_summer",   int(fd.month in [6, 7, 8])),
            ("covid", 0), ("post_covid", 0),
        ]:
            if feat in avail_feats:
                x_h[feat] = val
        if "t" in avail_feats and "t" in x_T.columns:
            new_t = float(x_T["t"].values[0] + h)
            x_h["t"] = new_t
            if "t_squared" in avail_feats:
                x_h["t_squared"] = new_t ** 2

        # user_exog для горизонта h
        ue_h = {
            col: vals[h - 1]
            for col, vals in prepared_user.items()
        } if prepared_user else None

        y_hat = float(fc_model.predict(x_h, h, cat=category, ch=channel, user_exog=ue_h)[0])
        y_hat = max(0.0, y_hat)

        # Приближённый ДИ: ±σ из исторических остатков (±20% как baseline)
        ci_width = y_hat * 0.20
        forecast_points.append({
            "date":     fd.strftime("%Y-%m-%d"),
            "forecast": round(y_hat, 0),
            "lower_80": round(max(0.0, y_hat - ci_width), 0),
            "upper_80": round(y_hat + ci_width, 0),
        })

    # MAPE из MLflow/CSV для отображения в UI
    ml_info  = _get_mlflow_mape(category, channel)
    test_mape = ml_info.get("no_exog")

    mode_used = (
        "full_exog"    if prepared_user and len(prepared_user) == len(selected_exog)
        else "partial_exog" if prepared_user
        else "no_exog"
    )

    return {
        "category":   category,
        "channel":    channel,
        "model_type": f"ML v2 XGBoost ({config})",
        "model_spec": (
            f"arch={meta.get('architecture','')}, "
            f"exog={selected_exog}, "
            f"mode={mode_used}"
        ),
        "model_class": "ml_v2",
        "exog_cols":   selected_exog,
        "test_mape":   test_mape,
        "mode":        mode_used,
        "horizon":     horizon,
        "forecast":    forecast_points,
    }