"""
агрегация результатов обучения
"""

from __future__ import annotations
import pandas as pd
from pathlib import Path
from src.config.settings import ML_RESULTS_DIR, RESULTS_DIR


def get_per_series_mape(category: str, channel: str,
                         variant: str = None) -> dict:

    from src.config.settings import ML_RESULTS_DIR
    results_dir = ML_RESULTS_DIR.parent/"part3"
    if variant is None:
        meta_path = results_dir/"lstm_prod_metadata.json"
        if meta_path.exists():
            import json
            with open(meta_path) as f:
                meta = json.load(f)
            variant = meta.get("prod_variant", "exog").replace("lstm_", "")
        else:
            variant = "exog"

    if variant.startswith("lstm_"):
        variant = variant[5:]
    csv_path = results_dir/f"lstm_{variant}_per_series.csv"
    if not csv_path.exists():
        return {"error": f"Per-series CSV не найден: {csv_path}"}

    df = pd.read_csv(csv_path)
    row = df[(df["category"] == category)&(df["channel"] == channel)]
#    if row.empty:
#        return {"error": f"Ряд '{category}|{channel}' не найден в per-series CSV"}

    r = row.iloc[0]
    return {"category":category,"channel":channel,"variant":variant,
        "mape_no_exog": float(r["mape_no_exog"]) if pd.notna(r.get("mape_no_exog")) else None,
        "mape_full_exog": float(r["mape_full_exog"]) if pd.notna(r.get("mape_full_exog")) else None}


def get_full_comparison_all_models(category: str, channel: str) -> dict:

    from src.config.settings import ML_RESULTS_DIR
    results_dir = ML_RESULTS_DIR.parent
    result = {
        "category": category,
        "channel": channel,
        "eco_mape": None,
        "eco_model": None,
        "ml_mape": None,
        "ml_model":None,
        "lstm_mape":None,
        "lstm_variant":None,
        "best_overall": None,
        "best_mape":None,
        "recommendation": ""}

    cross_path = results_dir/"part3"/"cross_model_comparison_per_series.csv"
    sarima_path = results_dir/"part1"/"sarima_all_models_comparison.csv"

    if cross_path.exists():
        df_cross = pd.read_csv(cross_path)
        r = df_cross[(df_cross["category"] == category)&(df_cross["channel"]  == channel)]
        if not r.empty:
            row = r.iloc[0]
            result["eco_mape"] = float(row["best_eco_mape"]) if pd.notna(row.get("best_eco_mape")) else None
            result["eco_model"] = str(row["best_eco_model"]) if pd.notna(row.get("best_eco_model")) else None
            result["ml_mape"] = float(row["best_ml_mape"]) if pd.notna(row.get("best_ml_mape")) else None
            result["ml_model"] = str(row["best_ml_model"]) if pd.notna(row.get("best_ml_model")) else None
            result["lstm_mape"] = float(row["lstm_mape"]) if pd.notna(row.get("lstm_mape")) else None
            result["lstm_variant"] = str(row["lstm_variant"]) if pd.notna(row.get("lstm_variant")) else None

    if result["eco_model"] is None and sarima_path.exists():
        df_s = pd.read_csv(sarima_path)
        for cc, hc in [("category","channel"), ("Категория","Канал")]:
            if cc in df_s.columns and hc in df_s.columns:
                rs = df_s[(df_s[cc] == category)&(df_s[hc] == channel)]
                if not rs.empty:
                    r2 = rs.iloc[0]
                    for lbl_col in ["Лучшая", "best_eco_model"]:
                        if lbl_col in r2.index and pd.notna(r2.get(lbl_col)):
                            result["eco_model"] = str(r2[lbl_col])
                            break
                    if result["eco_mape"] is None:
                        for mname, mcol in [("Naive","Naive MAPE,%"), ("ARIMA","ARIMA MAPE,%"),
                            ("SARIMA","SARIMA MAPE,%"), ("SARIMAX","SARIMAX MAPE,%"),
                            ("Prophet","Prophet MAPE,%")]:
                            if mcol in r2.index and pd.notna(r2.get(mcol)):
                                if result["eco_model"] == mname:
                                    result["eco_mape"] = float(r2[mcol])
                                    break
                    break

    ml_path = results_dir/"part2"/"test_metrics_h12.csv"
    if ml_path.exists():
        df_ml = pd.read_csv(ml_path)
        rows_ml = df_ml[(df_ml["category"] == category) & (df_ml["channel"] == channel)]
        if not rows_ml.empty:
            best_row = rows_ml.loc[rows_ml["mape"].idxmin()]
            result["ml_mape"] = round(float(best_row["mape"]), 2)
            result["ml_model"] = str(best_row.get("model", "ML"))

    lstm_ps = get_per_series_mape(category, channel)
    if "error" not in lstm_ps:
        result["lstm_mape"] = lstm_ps.get("mape_no_exog")
        result["lstm_variant"] = lstm_ps.get("variant")

    candidates = {}
    if result["eco_mape"] is not None: candidates["Эконометрика"] = result["eco_mape"]
    if result["ml_mape"] is not None: candidates["ML"] = result["ml_mape"]
    if result["lstm_mape"] is not None: candidates["LSTM"] = result["lstm_mape"]
    if candidates:
        best_label = min(candidates, key=candidates.get)
        best_mape = candidates[best_label]
        result["best_overall"] = best_label
        result["best_mape"] = round(best_mape,2)
        eco_m  = f"{result['eco_mape']:.1f}%" if result["eco_mape"] else "н/д"
        ml_m   = f"{result['ml_mape']:.1f}%" if result["ml_mape"] else "н/д"
        lstm_m = f"{result['lstm_mape']:.1f}%" if result["lstm_mape"] else "н/д"
        result["recommendation"] = (
            f"Для ряда {category}|{channel} рекомендуется: {best_label} "
            f"(MAPE={best_mape:.1f}%)"
            f"Эконометрика: {eco_m}, ML: {ml_m}, LSTM: {lstm_m}.")

    return result