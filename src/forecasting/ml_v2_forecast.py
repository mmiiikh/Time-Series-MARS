from __future__ import annotations
import os as _os
_os.environ.setdefault("OMP_NUM_THREADS","1")
_os.environ.setdefault("OPENBLAS_NUM_THREADS","1")
_os.environ.setdefault("MKL_NUM_THREADS","1")
_os.environ.setdefault("NUMEXPR_NUM_THREADS","1")
import json
import pickle
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from src.config.settings import ML_MODELS_DIR, ML_RESULTS_DIR, TARGET_COL, HORIZON
warnings.filterwarnings("ignore")
_GLOBAL_DF_CACHE: dict = {}
_MODEL_CACHE:dict = {}

def _get_global_df(lag_ranges: dict) -> pd.DataFrame:
    cache_key = str(sorted(lag_ranges.items()))
    if cache_key not in _GLOBAL_DF_CACHE:
        from src.data.load_data import load_data, create_series_dict
        from src.data.preprocess import build_global_dataset
        import src.config.settings as _cfg
        df_raw = load_data(str(_cfg.ML_DATA_FILE))
        sd = create_series_dict(df_raw)
        global_df, _, _, _ = build_global_dataset(sd, lag_ranges)
        _GLOBAL_DF_CACHE[cache_key] = global_df
        print(f"[ml_v2_forecast] global_df построен и закэширован ({len(global_df)} строк)")
    return _GLOBAL_DF_CACHE[cache_key]


def _get_model(config: str = None):
    if config is None:
        config = PROD_CONFIG
    if config not in _MODEL_CACHE:
        _MODEL_CACHE[config] = _load_model(config)
        print(f"[ml_v2_forecast] модель '{config}' загружена и закэширована")
    return _MODEL_CACHE[config]

PROD_CONFIG = "vif_exog"

def _load_metadata(config: str = PROD_CONFIG) -> dict:
    path = ML_MODELS_DIR / f"ml_v2_{config}_metadata.json"
    if not path.exists():
        raise FileNotFoundError(f"Метаданные ML v2 не найдены: {path}")
    with open(path) as f:
        return json.load(f)


def _load_model(config: str = PROD_CONFIG):
    meta = _load_metadata(config)
    pkl_path = Path(meta["winner_pkl_full"])
    if not pkl_path.exists():
        raise FileNotFoundError(f"pkl модели не найден: {pkl_path}")
    with open(pkl_path, "rb") as f:
        return pickle.load(f), meta


def _get_mlflow_mape(category: str, channel: str) -> dict:
    try:
        import mlflow
        from src.config.settings import MLFLOW_TRACKING_URI
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = mlflow.tracking.MlflowClient()
        exp = client.get_experiment_by_name("mars_ml_exog_v2_comparison")
        if exp is None:
            return {}
        runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            filter_string=f"tags.category = '{category}' and tags.channel = '{channel}'",
            max_results=10)
        for config_pref in [PROD_CONFIG, "all_exog"]:
            for run in runs:
                if run.data.tags.get("config_name") == config_pref:
                    m = run.data.metrics
                    no_exog = m.get("ml_v2_mape_no_exog")
                    partial = m.get("ml_v2_mape_partial_exog")
                    full = m.get("ml_v2_mape_full_exog")
                    return {
                        "no_exog": round(no_exog, 2) if no_exog else None,
                        "partial_exog": round(partial, 2) if partial else None,
                        "full_exog": round(full, 2)    if full    else None,
                        "config_name": config_pref,
                        "winner_model": run.data.tags.get("winner_v2_model", ""),
                        "architecture": run.data.tags.get("architecture", ""),
                        "selected_exog": run.data.tags.get("selected_exog", "")}
    except Exception:
        pass
    try:
        csv_path = ML_RESULTS_DIR / f"test_metrics_v2_{PROD_CONFIG}.csv"
        if not csv_path.exists():
            csv_path = ML_RESULTS_DIR / "test_metrics_v2.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            sub = df[(df["category"] == category)&(df["channel"] == channel)]
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
    try:
        import mlflow
        from src.config.settings import MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = mlflow.tracking.MlflowClient()
        for exp_name in [MLFLOW_EXPERIMENT, "mars-forecasting", "mars_sarima"]:
            exp = client.get_experiment_by_name(exp_name)
            if exp is None:
                continue
            runs = client.search_runs(experiment_ids=[exp.experiment_id], max_results=2000)
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
    sarima_mape = get_sarima_mape(category, channel)
    ml_info = _get_mlflow_mape(category, channel)
    ml_no_exog = ml_info.get("no_exog")
    best_model = None
    best_mape = None
    candidates = {}
    if sarima_mape is not None:
        candidates["sarima"] = sarima_mape
    if ml_no_exog is not None:
        candidates["ml_v2"] = ml_no_exog
    if candidates:
        best_model = min(candidates, key=candidates.get)
        best_mape = candidates[best_model]
    if best_model == "sarima":
        model_label = "SARIMA/SARIMAX"
        recommendation = (
            f"Для этого ряда рекомендуется {model_label} "
            f"(MAPE={sarima_mape:.1f}% на тесте)")
    elif best_model == "ml_v2":
        recommendation = (
            f"Для этого ряда рекомендуется ML v2 "
            f"(MAPE={ml_no_exog:.1f}% без экзогенных). "
            f"С плановыми данными MAPE может быть ниже.")
    else:
        recommendation = "Нет данных о качестве моделей."
    selected_exog: list[str] = []
    try:
        meta = _load_metadata(PROD_CONFIG)
        selected_exog = meta.get("selected_exog", [])
    except Exception:
        pass

    return {
        "sarima_mape":sarima_mape,
        "ml_v2_no_exog":ml_no_exog,
        "ml_v2_partial_exog": ml_info.get("partial_exog"),
        "ml_v2_full_exog": ml_info.get("full_exog"),
        "best_model": best_model,
        "best_mape": best_mape,
        "recommendation":recommendation,
        "selected_exog":selected_exog,
        "config_name":ml_info.get("config_name", PROD_CONFIG),
        "winner_v2_model": ml_info.get("winner_model", "")}


def forecast_ml_v2(
    series_dict: dict,
    category: str,
    channel: str,
    horizon: int,
    user_exog: dict[str, list[float]] | None = None,
    config: str = PROD_CONFIG) -> dict:
    fc_model, meta = _get_model(config)
    key = (category, channel)
    if key not in series_dict:
        raise ValueError(f"Ряд '{category}|{channel}' не найден")
    base_features = meta["base_features"]
    selected_exog = meta.get("selected_exog", [])
    last_value_per = {tuple(k.split("|")): v for k, v in meta.get("last_value_per_series", {}).items()}
    global_last = meta.get("global_last", {})
    fc_model.last_value_per_series = last_value_per
    fc_model.global_last = global_last
    lag_ranges = meta.get("ccf_lag_ranges", {})
    global_df = _get_global_df(lag_ranges)
    series_rows = global_df[(global_df["_category"] == category) &(global_df["_channel"]  == channel)]
    if series_rows.empty:
        raise ValueError(f"Нет данных в global_df для '{category}|{channel}'")

    avail_feats = [f for f in base_features if f in series_rows.columns]
    x_T = series_rows[avail_feats].fillna(0).iloc[[-1]].copy()
    last_date = pd.Timestamp(series_rows["_date"].max())
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
    for h in range(1, horizon+1):
        fd = last_date + pd.DateOffset(months=h)
        x_h = x_T.copy()
        for feat, val in [
            ("month", int(fd.month)),
            ("month_sin", float(np.sin(2*np.pi*fd.month/12))),
            ("month_cos", float(np.cos(2*np.pi*fd.month/12))),
            ("quarter", int(fd.quarter)),
            ("quarter_sin", float(np.sin(2*np.pi*fd.quarter/4))),
            ("quarter_cos", float(np.cos(2*np.pi*fd.quarter/4))),
            ("is_q4",int(fd.month >= 10)),
            ("is_summer", int(fd.month in [6,7,8])),
            ("covid", 0), ("post_covid", 0)]:
            if feat in avail_feats:
                x_h[feat] = val
        if "t" in avail_feats and "t" in x_T.columns:
            new_t = float(x_T["t"].values[0] + h)
            x_h["t"] = new_t
            if "t_squared" in avail_feats:
                x_h["t_squared"] = new_t ** 2

        ue_h = {col: vals[h - 1] for col, vals in prepared_user.items()} if prepared_user else None
        y_hat = float(fc_model.predict(x_h, h, cat=category, ch=channel, user_exog=ue_h)[0])
        y_hat = max(0.0, y_hat)
        ci_width = y_hat * 0.20
        forecast_points.append({
            "date": fd.strftime("%Y-%m-%d"),
            "forecast": round(y_hat, 0),
            "lower_80": round(max(0.0, y_hat - ci_width), 0),
            "upper_80": round(y_hat + ci_width, 0)})

    ml_info = _get_mlflow_mape(category, channel)
    test_mape = ml_info.get("no_exog")
    mode_used = ("full_exog" if prepared_user and len(prepared_user) == len(selected_exog)
        else "partial_exog" if prepared_user
        else "no_exog")

    return {
        "category": category,
        "channel": channel,
        "model_type": f"ML v2 XGBoost ({config})",
        "model_spec": (
            f"arch={meta.get('architecture','')}, "
            f"exog={selected_exog}, "
            f"mode={mode_used}"),
        "model_class": "ml_v2",
        "exog_cols": selected_exog,
        "test_mape": test_mape,
        "mode": mode_used,
        "horizon": horizon,
        "forecast": forecast_points}