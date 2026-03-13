import mlflow
import mlflow.sklearn
import numpy as np

from src.config.settings import (
    DATA_PATH, MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT,
    RESULTS_DIR,
)
from src.data.load_data import load_data, create_series_dict
from src.forecasting.sarima import (
    select_exog_variables,
    find_global_order,
    train_all_models,
    train_seasonal_naive,
    train_sarima_no_exog,
    select_best_and_save,
)


def run_training(use_individual_order: bool = True, test_size: int = 12):
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    print("обучение SARIMA/SARIMAX")

    df = load_data(str(DATA_PATH))
    series_dict = create_series_dict(df)


    print("\nОтбор экзогенных переменных")
    per_series_exog = select_exog_variables(series_dict)

    print("\nПодбор глобального порядка")
    global_order = find_global_order(series_dict, per_series_exog)

    print("\nОбучение SARIMAX")
    results_sarimax = train_all_models(
        series_dict, per_series_exog, global_order,
        use_individual_order=use_individual_order, test_size=test_size,
    )

    print("\nОбучение Naive")
    results_naive = train_seasonal_naive(series_dict, test_size=test_size)

    print("\nОбучение SARIMA(без экз)")
    results_sarima = train_sarima_no_exog(series_dict, global_order, test_size=test_size)

    print("\nВыбор лучшей модели")
    manifest = select_best_and_save(results_sarimax, results_sarima, results_naive)

    print("\nЛогирование в MLflow")
    _log_to_mlflow(manifest, results_sarimax, results_sarima, results_naive, global_order)

    print("\nОбучение завершено")
    return manifest


def _log_to_mlflow(manifest, results_sarimax, results_sarima, results_naive, global_order):
    all_results = {
        "Naive":  results_naive,
        "SARIMA":  results_sarima,
        "SARIMAX": results_sarimax}

    for key_str, info in manifest.items():
        cat = info["category"]
        ch  = info["channel"]
        key = (cat, ch)

        with mlflow.start_run(run_name=f"train|{cat}|{ch}"):
            mlflow.set_tag("log_type",  "training")
            mlflow.set_tag("category",  cat)
            mlflow.set_tag("channel",   ch)
            mlflow.set_tag("best_model", info["best_model"])
            mlflow.log_params({
                "best_model":     info["best_model"],
                "order":          str(info["order"]),
                "seasonal_order": str(info["seasonal_order"]),
                "exog_cols":      str(info["exog_cols"]),
                "global_p":       global_order["p"],
                "global_d":       global_order["d"],
                "global_q":       global_order["q"]})
            for model_name, results in all_results.items():
                if key not in results:
                    continue
                m = results[key]["metrics"]
                prefix = model_name.lower().replace("ï", "i").replace("'", "")
                for metric_name in ["mape", "smape", "mae", "rmse"]:
                    val = m.get(metric_name)
                    if val is not None and not (isinstance(val, float) and np.isnan(val)):
                        mlflow.log_metric(f"{prefix}_{metric_name}", round(float(val), 4))
            for metric_name, val in (info.get("metrics") or {}).items():
                if val is not None:
                    mlflow.log_metric(f"best_{metric_name}", round(float(val), 4))
            model_file = info.get("model_file")
            if model_file:
                mlflow.log_artifact(model_file, artifact_path="model")


if __name__ == "__main__":
    run_training()