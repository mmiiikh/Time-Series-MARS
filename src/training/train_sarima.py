import warnings
warnings.filterwarnings("ignore")
import mlflow
import numpy as np
import pandas as pd
from src.config.settings import (DATA_PATH, MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT,RESULTS_DIR)
from src.data.load_data import load_data, create_series_dict
from src.forecasting.sarima import (
    select_exog_variables,
    find_global_order,
    train_all_models,
    train_seasonal_naive,
    train_sarima_no_exog,
    train_arima,
    train_prophet,
    select_best_and_save,
    compare_all_models,
    SARIMAX_EXOG_CANDIDATES)


def run_training(use_individual_order: bool=True, test_size: int=12) -> dict:

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    df =load_data(str(DATA_PATH))
    series_dict = create_series_dict(df)
    per_series_exog = select_exog_variables(series_dict)
    global_order = find_global_order(series_dict, per_series_exog)
    results_sarimax = train_all_models(series_dict, per_series_exog, global_order,use_individual_order=use_individual_order, test_size=test_size)
    results_naive = train_seasonal_naive(series_dict, test_size=test_size)
    results_sarima = train_sarima_no_exog(series_dict, global_order,test_size=test_size)
    results_arima = train_arima(series_dict, test_size=test_size)
    results_prophet = train_prophet(series_dict,test_size=test_size)
    manifest = select_best_and_save(
        results_sarimax= results_sarimax,
        results_sarima = results_sarima,
        results_naive =results_naive,
        results_arima = results_arima,
        results_prophet = results_prophet)

    _log_to_mlflow(
        manifest = manifest,
        results_sarimax = results_sarimax,
        results_sarima= results_sarima,
        results_naive =results_naive,
        results_arima = results_arima,
        results_prophet = results_prophet,
        global_order = global_order)

    compare_all_models(
        results_sarimax = results_sarimax,
        results_sarima = results_sarima,
        results_naive =results_naive,
        results_arima = results_arima,
        results_prophet = results_prophet,
        save_csv = True)

    print(f"MLflow: mlflow ui --backend-store-uri {MLFLOW_TRACKING_URI}")
    return manifest


def _log_to_mlflow(
    manifest: dict,
    results_sarimax: dict,
    results_sarima:dict,
    results_naive: dict,
    results_arima:dict,
    results_prophet: dict,
    global_order: dict):

    all_results = {"naive":results_naive,"arima":results_arima,"sarima":results_sarima,"sarimax":results_sarimax,"prophet":results_prophet}
    logged=0
    for key_str, info in manifest.items():
        cat = info["category"]
        ch = info["channel"]
        key = (cat, ch)
        with mlflow.start_run(run_name=f"train|{cat}|{ch}"):
            mlflow.set_tag("log_type","training")
            mlflow.set_tag("category",cat)
            mlflow.set_tag("channel",ch)
            mlflow.set_tag("best_model",info["best_model"])
            mlflow.log_params({
                "best_model": info["best_model"],
                "order": str(info["order"]),
                "seasonal_order": str(info["seasonal_order"]),
                "exog_cols": str(info["exog_cols"]),
                "global_p": global_order["p"],
                "global_d": global_order["d"],
                "global_q": global_order["q"]})

            for prefix,results in all_results.items():
                if key not in results:
                    continue
                m = results[key].get("metrics", {})
                for metric_name in ["mape", "smape", "mae", "rmse"]:
                    val = m.get(metric_name)
                    if val is not None and not (isinstance(val, float) and np.isnan(val)):
                        mlflow.log_metric(f"{prefix}_{metric_name}",round(float(val), 4))

            for metric_name, val in (info.get("metrics") or {}).items():
                if val is not None:
                    mlflow.log_metric(f"best_{metric_name}", round(float(val), 4))

            model_file = info.get("model_file")
            if model_file:
                mlflow.log_artifact(model_file, artifact_path="model")
        logged += 1
    print(f"MLflow: записано {logged} runs,эксперимент {MLFLOW_EXPERIMENT}")


if __name__ == "__main__":
    run_training()