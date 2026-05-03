import warnings
warnings.filterwarnings("ignore")
from src.data.load_data import load_data, create_series_dict
from src.config.settings import DATA_PATH, RESULTS_DIR
from src.forecasting.sarima import train_arima, train_seasonal_naive, train_prophet, compare_all_models, load_manifest

def _rebuild_results_from_manifest(manifest: dict) -> tuple[dict, dict, dict, dict]:
    results_sarima = {}
    results_sarimax = {}
    results_naive = {}
    results_prophet = {}

    for key_str, info in manifest.items():
        cat  = info["category"]
        ch   = info["channel"]
        key  = (cat, ch)
        mape = info.get("mape")
        bm   = info.get("best_model", "")
        fake = {
            "metrics":{"mape": mape},
            "model_type": bm,
            "order": info.get("order", [0,0,0]),
            "seasonal_order": info.get("seasonal_order", [0,0,0,12]),
            "exog_cols": info.get("exog_cols", [])}
        if bm == "SARIMAX":
            results_sarimax[key] = fake
        elif bm == "SARIMA":
            results_sarima[key] = fake
        elif bm == "Naive":
            results_naive[key] = fake
        elif bm == "Prophet":
            results_prophet[key] = fake
    return results_sarima, results_sarimax, results_naive, results_prophet


def main():
    df = load_data(str(DATA_PATH))
    series_dict = create_series_dict(df)
    manifest = load_manifest()
    results_sarima, results_sarimax, results_naive, results_prophet = \
        _rebuild_results_from_manifest(manifest)
    results_arima = train_arima(series_dict, test_size=12)
    if not results_prophet:
        print("\nProphet не найден в манифесте,обучаем заново")
        results_prophet = train_prophet(series_dict, test_size=12)

    print("\nПересчёт Naive")
    results_naive_full = train_seasonal_naive(series_dict, test_size=12)
    results_naive_full.update(results_naive)
    results_naive = results_naive_full

    df_compare = compare_all_models(
        results_sarimax = results_sarimax,
        results_sarima = results_sarima,
        results_naive = results_naive,
        results_arima = results_arima,
        results_prophet = results_prophet,
        save_csv = True)
    return df_compare


if __name__ == "__main__":
    main()