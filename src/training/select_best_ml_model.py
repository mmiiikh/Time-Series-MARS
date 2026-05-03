import json
import pickle
import numpy as np
import pandas as pd
import mlflow

from src.config.settings import (
    ML_DATA_FILE, ML_MODELS_DIR, ML_RESULTS_DIR,
    MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_ML,
    TARGET_COL, CV_HORIZONS, HORIZON, RANDOM_STATE)
from src.data.load_data import load_data, create_series_dict
from src.data.preprocess import build_global_dataset, make_cv_folds
from src.forecasting.ml_model import DirectForecaster
from src.utils.metrics import compute_metrics

_MODEL_KEY = {
    "LightGBM": "lgbm",
    "XGBoost": "xgb",
    "RandomForest": "rf",
    "ElasticNet":  "en"}

def _model_key(model_name: str) -> str:
    return _MODEL_KEY.get(model_name, model_name.lower().replace(" ", "_"))


def get_best_model_from_mlflow() -> dict:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.tracking.MlflowClient()

    experiment = client.get_experiment_by_name(MLFLOW_EXPERIMENT_ML)
    if experiment is None:
        raise ValueError(
            f"Эксперимент '{MLFLOW_EXPERIMENT_ML}' не найден в MLflow. "
            f"Сначала запусти train_ml_experiments.py")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="tags.best_model = 'true'",
        order_by=["metrics.test_mape_h12_median ASC"])

    if not runs:
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["metrics.test_mape_h12_median ASC"])

    if not runs:
        raise ValueError("Нет runs в MLflow")

    best_run = runs[0]
    model_name = best_run.data.tags.get("model_type", "unknown")
    mape_h12= best_run.data.metrics.get("test_mape_h12_median", None)

    all_runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.test_mape_h12_median ASC"])
    for r in all_runs:
        name = r.data.tags.get("model_type", "?")
        m  = r.data.metrics.get("test_mape_h12_median", float("nan"))
        best = " best" if r.info.run_id == best_run.info.run_id else ""
        print(f"{name:<12}: {m:.2f}%{best}")

    return {"run_id": best_run.info.run_id,"model_name": model_name,"mape_h12":mape_h12}


def load_best_model(model_name: str) -> DirectForecaster:
    key = _model_key(model_name)
    pkl_path = ML_MODELS_DIR / f"ml_{key}_full.pkl"
    if not pkl_path.exists():
        raise FileNotFoundError(
            f"Файл {pkl_path} не найден. "
            f"Убедись что train_ml.py был запущен до конца.")

    with open(pkl_path, "rb") as f:
        fc = pickle.load(f)

    print(f"Загружена модель: {pkl_path.name}")
    return fc


def evaluate_per_series(fc: DirectForecaster,
                         model_name: str,
                         trained_eval: DirectForecaster,
                         series_dict: dict,
                         global_df: pd.DataFrame,
                         final_features: list) -> pd.DataFrame:
    from src.config.settings import TEST_SIZE
    dates = sorted(global_df["_date"].unique())
    train_dates = set(dates[:-TEST_SIZE])
    test_dates  = set(dates[-TEST_SIZE:])
    train_df = global_df[global_df["_date"].isin(train_dates)]
    test_df = global_df[global_df["_date"].isin(test_dates)]
    X_train = train_df[final_features].fillna(0)
    rows = []

    for (cat, ch), grp_te in test_df.groupby(["_category", "_channel"]):
        tr_s = train_df[(train_df["_category"] == cat) & (train_df["_channel"] == ch)]
        if tr_s.empty:
            continue
        x_T= X_train.loc[[tr_s.index[-1]]]
        te_dates = sorted(grp_te["_date"].unique())
        y_preds = []
        y_trues = []
        for h, future_date in enumerate(te_dates, start=1):
            fact_rows = grp_te[grp_te["_date"] == future_date]
            if fact_rows.empty or h not in trained_eval.models:
                continue
            x_h = x_T.copy()
            fd = pd.Timestamp(future_date)
            for feat, val in [
                ("month",int(fd.month)),
                ("month_sin",float(np.sin(2 * np.pi * fd.month/12))),
                ("month_cos", float(np.cos(2 * np.pi * fd.month/12))),
                ("quarter",int(fd.quarter)),
                ("quarter_sin", float(np.sin(2 * np.pi * fd.quarter/4))),
                ("quarter_cos", float(np.cos(2 * np.pi * fd.quarter/4))),
                ("is_q4", int(fd.month >= 10)),
                ("is_summer", int(fd.month in [6, 7, 8])),
                ("covid", 0),
                ("post_covid", 0)]:
                if feat in final_features:
                    x_h[feat] = val

            if "t" in final_features and "t" in x_T.columns:
                new_t = float(x_T["t"].values[0] + h)
                x_h["t"] = new_t
                if "t_squared" in final_features:
                    x_h["t_squared"] = new_t ** 2
            y_pred_h = float(trained_eval.predict(x_h, h)[0])
            y_true_h = float(np.expm1(fact_rows[TARGET_COL].values[0]))
            y_preds.append(y_pred_h)
            y_trues.append(y_true_h)

        if len(y_preds) == 0:
            continue
        m = compute_metrics(np.array(y_trues), np.array(y_preds))
        rows.append({
            "category":cat,
            "channel": ch,
            "ml_model": model_name,
            "n_horizons": len(y_preds),
            "horizon_type": "h1_12_from_T",**m})
    out = pd.DataFrame(rows).sort_values(["category", "channel"])
    return out


def get_best_part1_metrics_from_mlflow() -> pd.DataFrame:
    from src.config.settings import MLFLOW_EXPERIMENT
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(MLFLOW_EXPERIMENT)
    if experiment is None:
        for name in ["mars-forecasting", "mars_sarima", "mars_forecasting"]:
            experiment = client.get_experiment_by_name(name)
            if experiment is not None:
                print(f"Найден эксперимент: '{name}'")
                break

    if experiment is None:
        print("Эксперимент части 1 не найден в MLflow.")
        print("Доступные эксперименты:")
        for exp in client.search_experiments():
            print(f"{exp.name}")
        return pd.DataFrame()

    runs = client.search_runs(experiment_ids=[experiment.experiment_id],max_results=2000)
    if not runs:
        print("Нет runs в эксперименте части 1.")
        return pd.DataFrame()
    print(f"Найдено {len(runs)} runs в '{experiment.name}'")
    rows = []
    for run in runs:
        run_name = run.data.tags.get("mlflow.runName", "")
        parts = run_name.split("|")
        if len(parts) < 3:
            continue
        category = parts[1].strip()
        channel = parts[2].strip()
        params = run.data.params
        metrics = run.data.metrics
        best_model = params.get("best_model", "SARIMA")
        mape = metrics.get("best_mape")
        mae = metrics.get("best_mae")
        smape = metrics.get("best_smape")
        rmse = metrics.get("best_rmse")
        if mape is None:
            continue

        rows.append({
            "category": category,
            "channel": channel,
            "part1_model": best_model,
            "part1_mape": round(float(mape),  3),
            "part1_mae": round(float(mae),   0) if mae   else None,
            "part1_smape": round(float(smape), 3) if smape else None,
            "part1_rmse": round(float(rmse),  0) if rmse  else None})

    if not rows:
        print(" Не удалось извлечь метрики из runs части 1.")
        if runs:
            r = runs[0]
            print(f"Пример params:  {dict(list(r.data.params.items())[:6])}")
            print(f"Пример metrics: {dict(list(r.data.metrics.items())[:6])}")
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    best = (df.sort_values("part1_mape").groupby(["category", "channel"]).first().reset_index())
    return best

    if not rows:
        print("Не удалось извлечь метрики.")
        return pd.DataFrame()

    df   = pd.DataFrame(rows)
    best = df.sort_values("sarima_mape").groupby(["category", "channel"]).first().reset_index()
    print(f"Извлечено {len(best)} рядов")
    print(f"Медианный MAPE SARIMA: {best['sarima_mape'].median():.2f}%")
    return best


def compare_with_part1(ml_metrics: pd.DataFrame) -> pd.DataFrame:
    from src.config.settings import RESULTS_DIR
    part1_best = pd.DataFrame()

    for csv_name in ["test_metrics_sarima.csv", "sarima_metrics.csv","metrics.csv", "results.csv"]:
        csv_path = RESULTS_DIR / "part1" / csv_name
        if csv_path.exists():
            print(f"\n  Читаем метрики части 1 из {csv_path.name}...")
            df = pd.read_csv(csv_path)
            if "category" in df.columns and "channel" in df.columns \
                    and "mape" in df.columns:
                part1_best = (df.sort_values("mape")
                                .groupby(["category", "channel"])
                                .first()
                                .reset_index()
                                .rename(columns={
                                    "mape":"part1_mape",
                                    "mae":"part1_mae",
                                    "smape": "part1_smape",
                                    "model_type": "part1_model"}))
                break


    if part1_best.empty:
        print("CSV не найден. Читаем из MLflow...")
        part1_best = get_best_part1_metrics_from_mlflow()
    if part1_best.empty:
        print("Метрики части 1 недоступны.")
        return ml_metrics

    model_name = ml_metrics["ml_model"].iloc[0]
    compare = ml_metrics.merge(part1_best, on=["category", "channel"], how="left")

    compare["winner"] = compare.apply(
        lambda r: model_name
        if pd.notna(r.get("part1_mape")) and r["mape"] < r["part1_mape"]
        else r.get("part1_model", "Part1"),
        axis=1)
    compare["delta_mape"] = (compare["mape"] - compare["part1_mape"]).round(2)
    ml_wins = (compare["winner"] == model_name).sum()
    part1_wins = (compare["winner"] != model_name).sum()
    no_part1 = compare["part1_mape"].isna().sum()
    total = len(compare)

    if no_part1:
        print(f"Нет данных части 1: {no_part1} рядов")
    print(f"Медианный MAPE:")
    print(f"{model_name}: {compare['mape'].median():.2f}%")
    print(f"Победители по рядам:")
    print(compare["winner"].value_counts().to_string())
    return compare

MLFLOW_EXPERIMENT_COMPARISON = "mars_model_selection"


def log_per_series_to_mlflow(ml_metrics: pd.DataFrame,comparison: pd.DataFrame,best_info: dict):
    import re
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_COMPARISON)
    model_name = best_info["model_name"]
    has_part1 = "part1_mape" in comparison.columns
    key_model = _model_key(model_name)
    eval_pkl_path = str(ML_MODELS_DIR / f"ml_{key_model}_eval.pkl")
    full_pkl_path = str(ML_MODELS_DIR / f"ml_{key_model}_full.pkl")
    logged = 0
    for _, row in comparison.iterrows():
        cat = str(row["category"])
        ch = str(row["channel"])
        ml_mape = float(row["mape"])
        winner = str(row.get("winner", model_name))
        winner_type = "ML" if winner == model_name else "Part1"
        run_name = f"{cat}|{ch}"
        with mlflow.start_run(run_name=run_name):
            mlflow.set_tag("category", cat)
            mlflow.set_tag("channel", ch)
            mlflow.set_tag("winner",winner)
            mlflow.set_tag("winner_type",winner_type)
            mlflow.set_tag("ml_model_name", model_name)
            mlflow.set_tag("ml_pkl_full", full_pkl_path)
            mlflow.set_tag("ml_pkl_eval", eval_pkl_path)
            mlflow.set_tag("horizon_type","h1_12_from_T")

            ml_metrics_dict = {"ml_mape": round(ml_mape, 3)}
            if "mae" in row and pd.notna(row["mae"]):
                ml_metrics_dict["ml_mae"] = round(float(row["mae"]), 0)
            if "smape" in row and pd.notna(row.get("smape")):
                ml_metrics_dict["ml_smape"] = round(float(row["smape"]), 3)

            if has_part1 and pd.notna(row.get("part1_mape")):
                ml_metrics_dict["part1_mape"] = round(float(row["part1_mape"]), 3)
                ml_metrics_dict["delta_mape"] = round(ml_mape - float(row["part1_mape"]), 3)
                if pd.notna(row.get("part1_mae")):
                    ml_metrics_dict["part1_mae"] = round(float(row["part1_mae"]), 0)

            mlflow.log_metrics(ml_metrics_dict)
            mlflow.log_params({
                "evaluation": "h1_12_from_T",
                "part1_model": str(row.get("part1_model", "unknown"))})
        logged += 1

    with mlflow.start_run(run_name="_summary"):
        mlflow.set_tag("run_type", "summary")
        mlflow.set_tag("ml_model", model_name)

        mlflow.log_metrics({
            "total_series": len(comparison),
            "ml_median_mape": round(float(comparison["mape"].median()), 3),
            "ml_mean_mape":round(float(comparison["mape"].mean()), 3)})

        if has_part1:
            ml_wins = int((comparison["winner"] == model_name).sum())
            part1_wins = int((comparison["winner"] != model_name).sum())
            mlflow.log_metrics({
                "ml_wins": ml_wins,
                "part1_wins": part1_wins,
                "part1_median_mape":  round(float(comparison["part1_mape"].median()), 3),
                "median_delta_mape":  round(float(comparison["delta_mape"].median()), 3)})
            mlflow.set_tag("result",f"ML лучше в {ml_wins}/{len(comparison)} рядах")

        comp_path = ML_RESULTS_DIR / "ml_vs_part1_comparison.csv"
        ml_path = ML_RESULTS_DIR / "best_model_per_series.csv"
        if comp_path.exists():
            mlflow.log_artifact(str(comp_path))
        if ml_path.exists():
            mlflow.log_artifact(str(ml_path))

    print(f"Залогировано рядов: {logged}/{len(comparison)}")
    print(f"Эксперимент: {MLFLOW_EXPERIMENT_COMPARISON}")
    if has_part1:
        ml_wins = int((comparison["winner"] == model_name).sum())
        print(f"ML лучше в {ml_wins}/{len(comparison)} рядах")


def main():
    best_info  = get_best_model_from_mlflow()
    model_name = best_info["model_name"]
    meta_path = ML_MODELS_DIR / "ml_metadata.json"
    with open(meta_path) as f:
        meta = json.load(f)
    final_features = meta["final_features"]
    lag_ranges = meta["ccf_lag_ranges"]
    df_raw = load_data(str(ML_DATA_FILE))
    series_dict = create_series_dict(df_raw)
    global_df, _, _, _ = build_global_dataset(series_dict, lag_ranges)
    key = _model_key(model_name)
    eval_path = ML_MODELS_DIR / f"ml_{key}_eval.pkl"
    with open(eval_path, "rb") as f:
        trained_eval = pickle.load(f)
    print(f"Eval модель загружена: {eval_path.name}")
    ml_metrics = evaluate_per_series(
        fc=trained_eval,
        model_name=model_name,
        trained_eval=trained_eval,
        series_dict=series_dict,
        global_df=global_df,
        final_features=final_features)

    print(f"Медиана MAPE:{ml_metrics['mape'].median():.2f}%")
    print(f"Среднее MAPE:{ml_metrics['mape'].mean():.2f}%")
    print(f"Мин MAPE:{ml_metrics['mape'].min():.2f}%")
    print(f"Макс MAPE:{ml_metrics['mape'].max():.2f}%")
    print(f"Топ-5 лучших рядов:")
    print(ml_metrics.nsmallest(5, "mape")[["category", "channel", "mape", "mae"]].to_string(index=False))
    print(f"Топ-5 худших рядов:")
    print(ml_metrics.nlargest(5, "mape")[["category", "channel", "mape", "mae"]].to_string(index=False))

    ml_path = ML_RESULTS_DIR / "best_model_per_series.csv"
    ml_metrics.to_csv(ml_path, index=False)
    comparison = compare_with_part1(ml_metrics)
    comp_path = ML_RESULTS_DIR / "ml_vs_part1_comparison.csv"
    comparison.to_csv(comp_path, index=False)
    log_per_series_to_mlflow(ml_metrics, comparison, best_info)
    return ml_metrics, comparison


if __name__ == "__main__":
    main()