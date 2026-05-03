"""
lstm_forecast.py — инференс LSTM моделей для Mars прогнозирования.

МЕСТО: src/forecasting/lstm_forecast.py

ИСПОЛЬЗУЕТСЯ:
    routes.py — model_class="lstm" → forecast_lstm()
    app.py    — fetch_forecast_lstm()

ПОДДЕРЖИВАЕМЫЕ ВАРИАНТЫ:
    lstm_base       — LSTM без future экзогенных (только история)
    lstm_attn_base  — LSTM+Attention без future экзогенных
    lstm_exog       — LSTM с 5 VIF-экзогенными (no_exog / full_exog)
    lstm_attn_exog  — LSTM+Attention с 5 VIF-экзогенными

РЕЖИМЫ ИНФЕРЕНСА:
    no_exog   — future_exog = last_value_per_series (пользователь ничего не вводит)
    full_exog — future_exog = плановые значения от пользователя (5 переменных)

API функции:
    forecast_lstm(series_dict, category, channel, horizon, variant, user_exog)
      → dict (аналог forecast_from_manifest из sarima.py)

    get_lstm_info(variant) → dict
      → метаданные модели и тестовые MAPE

    get_lstm_comparison(category, channel) → dict
      → MAPE всех вариантов для ряда

CHECKPOINT СТРУКТУРА (из train_lstm.py step_save):
    {
      model_state_dict:  OrderedDict,
      model_config:      dict (input_size, n_categories, n_channels, ...),
      scaler_y:          SequenceScaler (MinMaxScaler для таргета),
      scaler_x:          SequenceScaler (MinMaxScaler для признаков),
      feature_names:     list[str],
      cat_to_id:         dict {category: int},
      ch_to_id:          dict {channel: int},
      last_value:        dict {"cat|ch": {col: float}},
      global_last:       dict {col: float},
      has_future_exog:   bool,
      window_size:       int (=24),
      horizon:           int (=12),
      vif_exog_vars:     list[str],
      future_exog_cols:  list[str],
      eval_summary:      dict {mape_no_exog_median, ...},
      variant_name:      str,
    }
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from functools import lru_cache
from pathlib import Path
from typing import Optional

from src.config.settings import TARGET_COL, SEASONAL_PERIOD, DL_DIR as _DL_DIR
from src.forecasting.lstm_model import (
    LSTMForecaster, LSTMAttentionForecaster,
    build_model_from_config,
    SequenceScaler,
    VIF_EXOG_VARS,
    FUTURE_EXOG_COLS,
    WINDOW_SIZE,
    HORIZON,
    safe_col)

try:
    torch.serialization.add_safe_globals([
        SequenceScaler,
        LSTMForecaster,
        LSTMAttentionForecaster])
except AttributeError:
    pass


VARIANT_PATHS = {
    "lstm_base": _DL_DIR / "lstm_base.pt",
    "lstm_attn_base": _DL_DIR / "lstm_attn_base.pt",
    "lstm_exog": _DL_DIR / "lstm_exog.pt",
    "lstm_attn_exog": _DL_DIR / "lstm_attn_exog.pt"}

PROD_VARIANT = "lstm_exog"


@lru_cache(maxsize=4)
def _load_checkpoint(variant: str) -> dict:
    path = VARIANT_PATHS.get(variant)
    if path is None:
        raise ValueError(
            f"Неизвестный вариант: '{variant}'. "
            f"Доступные: {list(VARIANT_PATHS.keys())}")
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint не найден: {path}. ")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = build_model_from_config(checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    checkpoint["_model"] = model
    sls_json = checkpoint.get("series_log_stats", {})
    checkpoint["_series_log_stats"] = {tuple(k.split("|", 1)): v for k, v in sls_json.items()}
    checkpoint["_global_mu"]  = float(checkpoint.get("global_mu",  0.0))
    checkpoint["_global_sig"] = float(checkpoint.get("global_sig", 1.0))
    if "cat_encoder" in checkpoint:
        checkpoint["cat_to_id"] = checkpoint["cat_encoder"]
    if "ch_encoder" in checkpoint:
        checkpoint["ch_to_id"] = checkpoint["ch_encoder"]

    lv_key = "last_value_per_series" if "last_value_per_series" in checkpoint else "last_value"
    lv_raw = checkpoint.get(lv_key, {})
    checkpoint["_last_value"] = {tuple(k.split("|", 1)): v for k, v in lv_raw.items()}

    return checkpoint


def _get_device() -> torch.device:
    return torch.device("cpu")


def _build_input_window(
    series_df:pd.DataFrame,
    feature_names: list[str],
    scaler_x:SequenceScaler,
    window_size:int,
    last_value: dict,
    global_last:dict,
    category:str,
    channel:str,
    has_future_exog:bool,
    user_exog:Optional[dict[str, float]] = None) -> torch.Tensor:
    avail = [f for f in feature_names if f in series_df.columns]
    missing_feats = [f for f in feature_names if f not in series_df.columns]

    df_window = series_df.iloc[-window_size:].copy()
    for f in missing_feats:
        df_window[f] = 0.0

    X = df_window[feature_names].values.copy().astype(np.float32)
    X = np.nan_to_num(X, nan=0.0)
    if has_future_exog:
        key = (category, channel)
        lv_series = last_value.get(key, {})

        for col in VIF_EXOG_VARS:
            fc = f"{safe_col(col)}_future"
            if fc not in feature_names:
                continue
            feat_idx = feature_names.index(fc)

            if user_exog is not None and col in user_exog:
                raw_val  = user_exog[col]
                if isinstance(raw_val, (list, tuple)):
                    fill_val = float(sum(raw_val) / len(raw_val)) if raw_val else 0.0
                else:
                    fill_val = float(raw_val)
            else:
                fill_val = lv_series.get(col, global_last.get(col, 0.0))

            X[-1, feat_idx] = fill_val

    X_scaled = scaler_x.transform(X)
    return torch.FloatTensor(X_scaled).unsqueeze(0)  # (1, window, features)


def _build_ci(y_pred:    np.ndarray,ci_width:  float = 0.20,) -> tuple[np.ndarray, np.ndarray]:
    lower = np.maximum(y_pred * (1-ci_width), 0.0)
    upper = y_pred * (1+ci_width)
    return lower, upper


def get_best_variant_for_series(category: str, channel: str) -> str:
    from src.config.settings import ML_RESULTS_DIR
    results_dir = ML_RESULTS_DIR.parent / "part3"
    best_variant = None
    best_mape = float("inf")
    for variant_name in ["base", "attn_base", "attn_exog", "exog"]:
        csv_path = results_dir / f"lstm_{variant_name}_per_series.csv"
        if not csv_path.exists():
            continue
        try:
            df = pd.read_csv(csv_path)
            row = df[(df["category"] == category) & (df["channel"] == channel)]
            if row.empty:
                continue
            mape = row.iloc[0].get("mape_no_exog")
            if mape is not None and pd.notna(mape) and float(mape) < best_mape:
                best_mape    = float(mape)
                best_variant = f"lstm_{variant_name}"
        except Exception:
            continue

    if best_variant is None:
        return PROD_VARIANT
    if not VARIANT_PATHS.get(best_variant, Path("")).exists():
        return PROD_VARIANT
    return best_variant



def forecast_lstm(
    series_dict: dict,
    category: str,
    channel: str,
    horizon: int = HORIZON,
    variant: str = PROD_VARIANT,
    user_exog:Optional[dict[str, float]] = None) -> dict:

    key = (category, channel)
    if key not in series_dict:
        raise ValueError(f"Ряд '{category}|{channel}' не найден в данных.")

    ckpt = _load_checkpoint(variant)
    model= ckpt["_model"]
    scaler_y = ckpt.get("scaler_y")
    scaler_x = ckpt["scaler_x"]
    feature_names = ckpt["feature_names"]
    cat_to_id = ckpt["cat_to_id"]
    ch_to_id = ckpt["ch_to_id"]
    last_value = ckpt["_last_value"]
    global_last = ckpt.get("global_last", {})
    has_future = ckpt.get("has_future_exog", False)
    window_size = ckpt.get("window_size", WINDOW_SIZE)
    eval_summary  = ckpt.get("eval_summary", {})
    variant_name  = ckpt.get("variant_name", variant)

    if category not in cat_to_id or channel not in ch_to_id:
        raise ValueError(
            f"Ряд '{category}|{channel}' не найден в энкодерах модели. "
            f"Модель обучалась на другом наборе рядов.")

    cat_id = cat_to_id[category]
    ch_id = ch_to_id[channel]

    series_df = _build_series_features_for_inference(series_dict[key], feature_names, last_value, global_last,category, channel)

    if len(series_df) < window_size:
        raise ValueError(
            f"Недостаточно данных для ряда '{category}|{channel}': "
            f"{len(series_df)} строк < window_size={window_size}")

    mode = "full_exog" if (has_future and user_exog) else "no_exog"
    exog_cols_used = ([col for col in VIF_EXOG_VARS if col in (user_exog or {})] if mode == "full_exog" else [])

    device = _get_device()
    X_tensor = _build_input_window(
        series_df, feature_names, scaler_x,
        window_size, last_value, global_last,
        category, channel, has_future, user_exog).to(device)
    cat_tensor = torch.LongTensor([cat_id]).to(device)
    ch_tensor = torch.LongTensor([ch_id]).to(device)

    model.to(device)
    with torch.no_grad():
        y_pred_scaled = model(X_tensor, cat_tensor, ch_tensor)
        y_pred_scaled = y_pred_scaled.cpu().numpy().ravel()  # (horizon,)

    sls = ckpt.get("_series_log_stats", {})
    g_mu = ckpt.get("_global_mu",  0.0)
    g_sig = ckpt.get("_global_sig", 1.0)
    stat = sls.get((category, channel), {"mean": g_mu, "std": g_sig})

    scaler_y = ckpt.get("scaler_y")

    if not sls and scaler_y is not None:
        import warnings
        warnings.warn(
            f"Checkpoint {variant} устаревший (без per-series нормализации). "
            "Переобучите: python -m src.training.train_lstm",
            UserWarning, stacklevel=2)

    if sls:
        y_pred_log = y_pred_scaled * stat["std"] + stat["mean"]
        y_pred = np.maximum(np.expm1(y_pred_log), 0.0)
    elif scaler_y is not None:
        y_pred_double_log = scaler_y.inverse_transform(y_pred_scaled)
        y_pred_log = np.expm1(y_pred_double_log)
        y_pred = np.maximum(np.expm1(y_pred_log), 0.0)
    else:
        y_pred = np.maximum(np.expm1(y_pred_scaled), 0.0)

    h_actual = min(horizon, HORIZON, len(y_pred))
    y_pred = y_pred[:h_actual]

    lower, upper = _build_ci(y_pred)

    y_series = series_dict[key][TARGET_COL].dropna()
    last_date = y_series.index[-1]
    future_idx = pd.date_range(start=pd.Timestamp(last_date) + pd.DateOffset(months=1),periods=h_actual,freq="MS")

    forecast_points = [
        {
            "date": d.strftime("%Y-%m-%d"),
            "forecast": round(float(v), 2),
            "lower_80": round(float(lo), 2),
            "upper_80": round(float(hi), 2)}
        for d, v, lo, hi in zip(future_idx, y_pred, lower, upper)]

    test_mape = eval_summary.get("mape_full_exog_median" if mode == "full_exog" else "mape_no_exog_median")

    model_class_name = ckpt["model_config"].get("model_class", "LSTMForecaster")
    model_type = {
        "lstm_base": "LSTM (base)",
        "lstm_attn_base": "LSTM + Attention (base)",
        "lstm_exog": "LSTM (exog)",
        "lstm_attn_exog": "LSTM + Attention (exog)"}.get(variant, variant)

    return {
        "category": category,
        "channel": channel,
        "model_type": model_type,
        "model_spec": (
            f"{model_class_name} | window={window_size} | "
            f"hidden={ckpt['model_config'].get('hidden_size', '?')} | "
            f"mode={mode}"),
        "model_class": "lstm",
        "variant": variant,
        "mode": mode,
        "exog_cols": exog_cols_used,
        "test_mape": round(float(test_mape), 2) if test_mape is not None else None,
        "horizon": h_actual,
        "forecast": forecast_points}

def _build_series_features_for_inference(
    df: pd.DataFrame,
    feature_names: list[str],
    last_value: dict,
    global_last: dict,
    category: str,
    channel: str) -> pd.DataFrame:

    y = df[TARGET_COL].dropna()
    out = pd.DataFrame(index=y.index)

    out["y_log"] = np.log1p(y.values)

    for lag in [1, 2, 3, 6, 12]:
        out[f"y_lag{lag}"] = out["y_log"].shift(lag)

    out["rolling_mean_3"] = out["y_log"].rolling(3).mean()
    out["rolling_mean_6"] = out["y_log"].rolling(6).mean()
    out["rolling_std_3"] = out["y_log"].rolling(3).std()
    out["rolling_min_6"] = out["y_log"].rolling(6).min()
    out["ewm_03"] = out["y_log"].ewm(alpha=0.3, adjust=False).mean()
    out["ewm_07"] = out["y_log"].ewm(alpha=0.7, adjust=False).mean()

    idx = pd.DatetimeIndex(y.index)
    out["month_sin"] = np.sin(2 * np.pi * idx.month / 12)
    out["month_cos"] = np.cos(2 * np.pi * idx.month / 12)
    out["quarter_sin"] = np.sin(2 * np.pi * idx.quarter / 4)
    out["quarter_cos"] = np.cos(2 * np.pi * idx.quarter / 4)
    out["is_q4"] = (idx.month >= 10).astype(float)
    out["is_summer"] = idx.month.isin([6, 7, 8]).astype(float)

    t = np.arange(len(y), dtype=float)
    out["t"] = t
    out["t_squared"] = t**2

    covid_dates = pd.date_range("2020-03-01", "2020-06-01", freq="MS")
    out["covid"] = idx.isin(covid_dates).astype(float)
    out["post_covid"] = (idx > pd.Timestamp("2020-06-01")).astype(float)

    key = (category, channel)
    lv = last_value.get(key, {})
    for col in VIF_EXOG_VARS:
        sc = safe_col(col)
        if col in df.columns:
            x = df[col].reindex(y.index)
            out[f"{sc}_lag1"] = x.shift(1).fillna(lv.get(col, global_last.get(col, 0.0)))
            out[f"{sc}_lag2"] = x.shift(2).fillna(lv.get(col, global_last.get(col, 0.0)))
        else:
            fill = lv.get(col, global_last.get(col, 0.0))
            out[f"{sc}_lag1"] = fill
            out[f"{sc}_lag2"] = fill

    for col in VIF_EXOG_VARS:
        fc = f"{safe_col(col)}_future"
        if fc in feature_names:
            out[fc] = lv.get(col, global_last.get(col, 0.0))

    for f in feature_names:
        if f not in out.columns and f not in ("y_log",):
            out[f] = 0.0

    return out.reset_index(drop=True)


def get_lstm_info(variant: str = PROD_VARIANT) -> dict:
    try:
        ckpt = _load_checkpoint(variant)
    except (FileNotFoundError, ValueError) as e:
        return {"error": str(e), "variant": variant}

    eval_summary = ckpt.get("eval_summary", {})
    config = ckpt.get("model_config", {})

    return {
        "variant": variant,
        "model_class": config.get("model_class", "?"),
        "has_future_exog": ckpt.get("has_future_exog", False),
        "vif_exog_vars":ckpt.get("vif_exog_vars", VIF_EXOG_VARS),
        "window_size": ckpt.get("window_size", WINDOW_SIZE),
        "horizon": ckpt.get("horizon", HORIZON),
        "hidden_size": config.get("hidden_size"),
        "n_layers": config.get("n_layers"),
        "n_features": len(ckpt.get("feature_names", [])),
        "mape_no_exog": eval_summary.get("mape_no_exog_median"),
        "mape_full_exog": eval_summary.get("mape_full_exog_median")}


def get_lstm_comparison(category: str, channel: str) -> dict:
    result = {"category": category,"channel":  channel,"variants": {}}
    for variant in VARIANT_PATHS:
        path = VARIANT_PATHS[variant]
        if not path.exists():
            continue
        try:
            ckpt = _load_checkpoint(variant)
            summary = ckpt.get("eval_summary", {})
            result["variants"][variant] = {
                "mape_no_exog":  summary.get("mape_no_exog_median"),
                "mape_full_exog": summary.get("mape_full_exog_median"),
                "has_future_exog": ckpt.get("has_future_exog", False)}
        except Exception:
            continue

    best_variant = None
    best_mape = float("inf")
    for v, info in result["variants"].items():
        mape = info.get("mape_no_exog")
        if mape is not None and mape < best_mape:
            best_mape = mape
            best_variant = v
    result["best_variant"] = best_variant
    result["best_mape"] = best_mape if best_variant else None

    return result


def is_lstm_available() -> bool:
    return any(p.exists() for p in VARIANT_PATHS.values())


def list_available_variants() -> list[str]:
    return [v for v, p in VARIANT_PATHS.items() if p.exists()]


def is_lstm_available() -> bool:
    return any(p.exists() for p in VARIANT_PATHS.values())


from src.forecasting.comparison import (   # noqa: E402
    get_per_series_mape,
    get_full_comparison_all_models,
)