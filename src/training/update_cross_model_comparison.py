from __future__ import annotations
import warnings
warnings.filterwarnings("ignore")
import json
import numpy as np
import pandas as pd
from pathlib import Path
import mlflow
from src.config.settings import MLFLOW_TRACKING_URI, ML_RESULTS_DIR, RESULTS_DIR

RESULTS1 = RESULTS_DIR / "part1"
RESULTS2 = ML_RESULTS_DIR
RESULTS3 = ML_RESULTS_DIR.parent / "part3"
OUTPUT_CSV = RESULTS3 / "cross_model_comparison_per_series.csv"
SARIMA_CSV = RESULTS1 / "sarima_all_models_comparison.csv"
ML_CSV = RESULTS2 / "test_metrics_v2_vif_exog.csv"

LSTM_CSVS  = [
    RESULTS3 / "lstm_exog_per_series.csv",
    RESULTS3 / "lstm_attn_exog_per_series.csv",
    RESULTS3 / "lstm_base_per_series.csv"]

MLFLOW_EXP = "mars_cross_model_update"


def load_sarima(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError()
    df = pd.read_csv(path)
    rename = {"Категория": "category","Канал": "channel","Лучшая": "best_eco_model_raw"}
    df = df.rename(columns=rename)

    mape_rename = {
        "Naive MAPE,%": "naive_mape",
        "ARIMA MAPE,%": "arima_mape",
        "SARIMA MAPE,%":"sarima_mape",
        "SARIMAX MAPE,%": "sarimax_mape",
        "Prophet MAPE,%": "prophet_mape"}
    df = df.rename(columns={k: v for k, v in mape_rename.items() if k in df.columns})

    eco_mape_cols = ["naive_mape", "arima_mape", "sarima_mape","sarimax_mape", "prophet_mape"]
    rows = []
    for _, r in df.iterrows():
        cat = str(r.get("category", "")).strip()
        ch  = str(r.get("channel",  "")).strip()
        if not cat or not ch:
            continue
        mapes = {}
        for col in eco_mape_cols:
            if col in r.index and pd.notna(r[col]):
                mapes[col] = float(r[col])

        model_name_map = {
            "naive_mape":"Naive",
            "arima_mape": "ARIMA",
            "sarima_mape":"SARIMA",
            "sarimax_mape":"SARIMAX",
            "prophet_mape": "Prophet"}
        if mapes:
            best_col = min(mapes, key=mapes.get)
            best_model = model_name_map[best_col]
            best_mape = mapes[best_col]
        else:
            best_model, best_mape = None, None

        raw_best = str(r.get("best_eco_model_raw", "")).strip()
        if raw_best and raw_best not in ("—", "nan", ""):
            best_model = raw_best

        rows.append({
            "category": cat,
            "channel": ch,
            "best_eco_model": best_model,
            "best_eco_mape": round(best_mape, 2) if best_mape is not None else None,
            **{col: round(mapes[col], 2) if col in mapes else None for col in eco_mape_cols}})

    result = pd.DataFrame(rows)
    print(f"Лучшие модели: {result['best_eco_model'].value_counts().to_dict()}")
    return result


def load_ml(path: Path) -> pd.DataFrame:
    if not path.exists():
        fallback = path.parent / "test_metrics_h12.csv"
        if fallback.exists():
            print(f"{path.name} не найден, fallback на {fallback.name}")
            path = fallback
        else:
            print(f"ML: файл не найден")
            return pd.DataFrame(columns=["category","channel","best_ml_mape","best_ml_model"])

    df = pd.read_csv(path)
    print(f"ML CSV [{path.name}]: {df.shape}")
    prod_model = "XGBoost"
    df_prod = df.copy()
    if "model" in df.columns:
        df_xgb = df[df["model"] == prod_model]
        if not df_xgb.empty:
            df_prod = df_xgb
            print(f"ML: model={prod_model},{len(df_prod)} строк")
        else:
            print(f"{prod_model} не найден, используем все модели")

    if "mode" in df_prod.columns:
        df_no = df_prod[df_prod["mode"] == "no_exog"]
        if not df_no.empty:
            df_prod = df_no
            print(f"ML: mode=no_exog, {len(df_prod)} строк")

    if "mape" not in df_prod.columns:
        print("ML: колонка mape не найдена")
        return pd.DataFrame(columns=["category","channel","best_ml_mape","best_ml_model"])
    rows = []
    for _, r in df_prod.iterrows():
        cat = str(r.get("category","")).strip()
        ch  = str(r.get("channel", "")).strip()
        if not cat or not ch:
            continue
        mape = float(r["mape"]) if pd.notna(r.get("mape")) else None
        rows.append({
            "category":cat,
            "channel":ch,
            "best_ml_mape":round(mape,2) if mape is not None else None,
            "best_ml_model": prod_model})

    df_out = pd.DataFrame(rows)
    df_out = df_out[df_out["category"] != ""]
    if df_out.duplicated(subset=["category","channel"]).any():
        n = len(df_out)
        df_out = (df_out.sort_values("best_ml_mape").drop_duplicates(subset=["category","channel"], keep="first").reset_index(drop=True))
        print(f"ML: дедупликация {n}, {len(df_out)}")
    return df_out


def load_lstm(csv_list: list[Path]) -> pd.DataFrame:
    path = None
    for p in csv_list:
        if p.exists():
            path = p
            break
    if path is None:
        print(f"LSTM: ни один CSV не найден")
        return pd.DataFrame(columns=["category", "channel","lstm_mape", "lstm_variant"])
    df = pd.read_csv(path)
    variant_name = path.stem.replace("_per_series", "").replace("lstm_", "lstm_")
    mape_col = None
    for c in ["mape_no_exog", "lstm_mape_no_exog", "mape", "lstm_mape"]:
        if c in df.columns:
            mape_col = c
            break
    if mape_col is None:
        print(f"LSTM: колонка MAPE не найдена в {path.name}")
        return pd.DataFrame(columns=["category", "channel","lstm_mape", "lstm_variant"])
    result = pd.DataFrame({
        "category": df["category"].astype(str).str.strip(),
        "channel": df["channel"].astype(str).str.strip(),
        "lstm_mape": df[mape_col].round(2),
        "lstm_variant": df.get("variant", pd.Series([variant_name] * len(df)))})
    return result


def build_comparison(
    df_eco:  pd.DataFrame,
    df_ml:   pd.DataFrame,
    df_lstm: pd.DataFrame) -> pd.DataFrame:

    df = df_eco.merge(df_ml,on=["category", "channel"], how="outer")
    df = df.merge(df_lstm, on=["category", "channel"], how="outer")
    winners = []
    winner_models = []
    winner_mapes = []
    for _, row in df.iterrows():
        candidates = {}
        if pd.notna(row.get("best_eco_mape")):
            candidates["econometric"] = float(row["best_eco_mape"])
        if pd.notna(row.get("best_ml_mape")):
            candidates["ml_v2"]  = float(row["best_ml_mape"])
        if pd.notna(row.get("lstm_mape")):
            candidates["lstm"]   = float(row["lstm_mape"])
        if not candidates:
            winners.append(None)
            winner_models.append(None)
            winner_mapes.append(None)
            continue
        best_class = min(candidates, key=candidates.get)
        best_mape = candidates[best_class]
        if best_class == "econometric":
            best_model = str(row.get("best_eco_model") or "SARIMA")
        elif best_class == "ml_v2":
            best_model = str(row.get("best_ml_model") or "ML")
        else:
            best_model = str(row.get("lstm_variant") or "lstm_exog")
        winners.append(best_class)
        winner_models.append(best_model)
        winner_mapes.append(round(best_mape, 2))
    df["winner"] = winners
    df["winner_model"] = winner_models
    df["winner_mape"] = winner_mapes
    return df

def print_summary(df: pd.DataFrame) -> None:
    class_info = [
        ("econometric","Эконометрика","best_eco_mape"),
        ("ml_v2", "ML","best_ml_mape"),
        ("lstm","LSTM","lstm_mape")]
    for cls, label, mape_col in class_info:
        if mape_col not in df.columns:
            continue
        med  = df[mape_col].dropna().median()
        wins = int((df["winner"] == cls).sum())
        print(f"{label:<20}{med:>11.2f}%{wins:>12}")
    if "winner_model" in df.columns:
        for model, cnt in df["winner_model"].value_counts().items():
            cls = df[df["winner_model"] == model]["winner"].iloc[0] if cnt > 0 else "—"
            cls_label = {"econometric": "Эконометрика","ml_v2": "ML", "lstm": "LSTM"}.get(cls, cls)


def log_to_mlflow(df: pd.DataFrame) -> None:
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXP)
        with mlflow.start_run(run_name="cross_model_update"):
            mlflow.set_tag("ml_retrained","False")
            mlflow.set_tag("lstm_retrained","False")
            mlflow.set_tag("sarima_retrained","True")
            for col, name in [
                ("best_eco_mape", "eco_mape_median"),
                ("best_ml_mape","ml_mape_median"),
                ("lstm_mape", "lstm_mape_median")]:
                if col in df.columns:
                    med = df[col].dropna().median()
                    if pd.notna(med):
                        mlflow.log_metric(name, round(float(med), 3))
            for cls, label in [("econometric","eco"),("ml_v2","ml"),("lstm","lstm")]:
                wins = int((df["winner"] == cls).sum())
                mlflow.log_metric(f"wins_{label}", wins)
            mlflow.log_metric("total_series", len(df))
        print(f"MLflow {MLFLOW_EXP}")
    except Exception as e:
        print(f"MLflow: {e}")

def main() -> pd.DataFrame:
    df_eco = load_sarima(SARIMA_CSV)
    df_ml = load_ml(ML_CSV)
    df_lstm = load_lstm(LSTM_CSVS)
    df = build_comparison(df_eco, df_ml, df_lstm)
    print_summary(df)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV,index=False)
    log_to_mlflow(df)
    return df


if __name__ == "__main__":
    main()