"""
Выбор лучшей ML модели и сохранение предсказаний по каждому ряду.
Используется для сравнения ML vs SARIMA по парам (category, channel).

Запуск:
    python -m src.training.select_best_ml_model

Что делает:
    1. Читает все runs из MLflow (mars_ml_models)
    2. Выбирает лучшую модель по test_mape_h12_median
       (сопоставимо с SARIMA — прогноз из одной точки на 12 мес.)
    3. Загружает модель и считает метрики по каждому ряду
    4. Сохраняет results/part2/best_model_per_series.csv
       — этот файл потом объединяется с SARIMA метриками

Формат выходного файла:
    category | channel | ml_model | mape | mae | rmse | smape |
    horizon_type = "h1..12_from_T" (как SARIMA)
"""

import json
import pickle
import numpy as np
import pandas as pd
import mlflow

from src.config.settings import (
    ML_DATA_FILE, ML_MODELS_DIR, ML_RESULTS_DIR,
    MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_ML,
    TARGET_COL, CV_HORIZONS, HORIZON, RANDOM_STATE,
)
from src.data.load_data import load_data, create_series_dict
from src.data.preprocess import build_global_dataset, make_cv_folds
from src.forecasting.ml_model import DirectForecaster
from src.utils.metrics import compute_metrics


# =============================================================================
# ШАГ 1: ВЫБОР ЛУЧШЕЙ МОДЕЛИ ИЗ MLFLOW
# =============================================================================

def get_best_model_from_mlflow() -> dict:
    """
    Читает все runs из эксперимента mars_ml_models.
    Возвращает run с минимальным test_mape_h12_median.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.tracking.MlflowClient()

    # Находим эксперимент
    experiment = client.get_experiment_by_name(MLFLOW_EXPERIMENT_ML)
    if experiment is None:
        raise ValueError(
            f"Эксперимент '{MLFLOW_EXPERIMENT_ML}' не найден в MLflow. "
            f"Сначала запусти train_ml_experiments.py"
        )

    # Все runs из эксперимента
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="tags.best_model = 'true'",
        order_by=["metrics.test_mape_h12_median ASC"],
    )

    if not runs:
        # Если тег best_model не выставлен — берём по метрике напрямую
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["metrics.test_mape_h12_median ASC"],
        )

    if not runs:
        raise ValueError("Нет runs в MLflow. Запусти train_ml.py сначала.")

    best_run = runs[0]
    model_name = best_run.data.tags.get("model_type", "unknown")
    mape_h12   = best_run.data.metrics.get("test_mape_h12_median", None)

    print(f"\n=== ЛУЧШАЯ ML МОДЕЛЬ ===")
    print(f"  Модель:              {model_name}")
    print(f"  Run ID:              {best_run.info.run_id}")
    print(f"  test_mape_h12_median: {mape_h12:.2f}%" if mape_h12 else "")

    # Выводим все runs для сравнения
    all_runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.test_mape_h12_median ASC"],
    )
    print("\n  Все модели (по test_mape_h12_median):")
    for r in all_runs:
        name = r.data.tags.get("model_type", "?")
        m    = r.data.metrics.get("test_mape_h12_median", float("nan"))
        best = " ← ЛУЧШАЯ" if r.info.run_id == best_run.info.run_id else ""
        print(f"    {name:<12}: {m:.2f}%{best}")

    return {
        "run_id":     best_run.info.run_id,
        "model_name": model_name,
        "mape_h12":   mape_h12,
    }


# =============================================================================
# ШАГ 2: ЗАГРУЗКА МОДЕЛИ
# =============================================================================

def load_best_model(model_name: str) -> DirectForecaster:
    """
    Загружает pkl файл лучшей модели (full версия — обучена на всех данных).
    """
    key      = "lgbm" if model_name == "LightGBM" else "xgb"
    pkl_path = ML_MODELS_DIR / f"ml_{key}_full.pkl"

    if not pkl_path.exists():
        raise FileNotFoundError(
            f"Файл {pkl_path} не найден. "
            f"Убедись что train_ml.py был запущен до конца."
        )

    with open(pkl_path, "rb") as f:
        fc = pickle.load(f)

    print(f"\n  Загружена модель: {pkl_path.name}")
    print(f"  Горизонты: h=1..{max(fc.models.keys())}")
    return fc


# =============================================================================
# ШАГ 3: МЕТРИКИ ПО КАЖДОМУ РЯДУ (h=1..12 из одной точки, как SARIMA)
# =============================================================================

def evaluate_per_series(fc: DirectForecaster,
                         model_name: str,
                         trained_eval: DirectForecaster,
                         series_dict: dict,
                         global_df: pd.DataFrame,
                         final_features: list) -> pd.DataFrame:
    """
    Считает метрики для каждой пары (category, channel).

    Логика (как SARIMA):
      - Стартовая точка T = последняя строка train (до теста)
      - Предсказываем h=1..12 вперёд из T, обновляем только календарь
      - Сравниваем с реальными значениями тестового периода

    Используем trained_eval (не full) — он не видел тест → честная оценка.
    """
    from src.config.settings import TEST_SIZE

    dates       = sorted(global_df["_date"].unique())
    train_dates = set(dates[:-TEST_SIZE])
    test_dates  = set(dates[-TEST_SIZE:])

    train_df = global_df[global_df["_date"].isin(train_dates)]
    test_df  = global_df[global_df["_date"].isin(test_dates)]
    X_train  = train_df[final_features].fillna(0)

    rows = []

    for (cat, ch), grp_te in test_df.groupby(["_category", "_channel"]):
        # Стартовая точка T = последняя строка train этого ряда
        tr_s = train_df[
            (train_df["_category"] == cat) & (train_df["_channel"] == ch)
        ]
        if tr_s.empty:
            continue

        x_T      = X_train.loc[[tr_s.index[-1]]]
        te_dates = sorted(grp_te["_date"].unique())
        y_preds  = []
        y_trues  = []

        for h, future_date in enumerate(te_dates, start=1):
            fact_rows = grp_te[grp_te["_date"] == future_date]
            if fact_rows.empty or h not in trained_eval.models:
                continue

            x_h = x_T.copy()
            fd  = pd.Timestamp(future_date)

            for feat, val in [
                ("month",       int(fd.month)),
                ("month_sin",   float(np.sin(2 * np.pi * fd.month / 12))),
                ("month_cos",   float(np.cos(2 * np.pi * fd.month / 12))),
                ("quarter",     int(fd.quarter)),
                ("quarter_sin", float(np.sin(2 * np.pi * fd.quarter / 4))),
                ("quarter_cos", float(np.cos(2 * np.pi * fd.quarter / 4))),
                ("is_q4",       int(fd.month >= 10)),
                ("is_summer",   int(fd.month in [6, 7, 8])),
                ("covid",       0),
                ("post_covid",  0),
            ]:
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
            "category":    cat,
            "channel":     ch,
            "ml_model":    model_name,
            "n_horizons":  len(y_preds),
            "horizon_type": "h1_12_from_T",
            **m,
        })

    out = pd.DataFrame(rows).sort_values(["category", "channel"])
    return out


# =============================================================================
# ШАГ 4: СРАВНЕНИЕ С SARIMA — читаем метрики из MLflow
# =============================================================================

def get_best_part1_metrics_from_mlflow() -> pd.DataFrame:
    """
    Читает метрики лучшей модели части 1 из MLflow (mars-forecasting).
    Лучшая модель для каждого ряда — SARIMA, SARIMAX или Naive.

    Формат run name: "train|{category}|{channel}"
    Params:  best_model (SARIMA / SARIMAX / Naive)
    Metrics: best_mape, best_mae, best_smape, best_rmse
    """
    from src.config.settings import MLFLOW_EXPERIMENT

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.tracking.MlflowClient()

    experiment = client.get_experiment_by_name(MLFLOW_EXPERIMENT)
    if experiment is None:
        for name in ["mars-forecasting", "mars_sarima", "mars_forecasting"]:
            experiment = client.get_experiment_by_name(name)
            if experiment is not None:
                print(f"  Найден эксперимент: '{name}'")
                break

    if experiment is None:
        print("  [WARN] Эксперимент части 1 не найден в MLflow.")
        print("  Доступные эксперименты:")
        for exp in client.search_experiments():
            print(f"    - {exp.name}")
        return pd.DataFrame()

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        max_results=2000,
    )

    if not runs:
        print("  [WARN] Нет runs в эксперименте части 1.")
        return pd.DataFrame()

    print(f"  Найдено {len(runs)} runs в '{experiment.name}'")

    rows = []
    for run in runs:
        # category и channel из runName: "train|CATEGORY|CHANNEL"
        run_name = run.data.tags.get("mlflow.runName", "")
        parts    = run_name.split("|")
        if len(parts) < 3:
            continue

        category = parts[1].strip()
        channel  = parts[2].strip()

        params  = run.data.params
        metrics = run.data.metrics

        # Лучшая модель для этого ряда
        best_model = params.get("best_model", "SARIMA")

        # Метрики лучшей модели
        mape  = metrics.get("best_mape")
        mae   = metrics.get("best_mae")
        smape = metrics.get("best_smape")
        rmse  = metrics.get("best_rmse")

        if mape is None:
            continue

        rows.append({
            "category":   category,
            "channel":    channel,
            "part1_model": best_model,
            "part1_mape":  round(float(mape),  3),
            "part1_mae":   round(float(mae),   0) if mae   else None,
            "part1_smape": round(float(smape), 3) if smape else None,
            "part1_rmse":  round(float(rmse),  0) if rmse  else None,
        })

    if not rows:
        print("  [WARN] Не удалось извлечь метрики из runs части 1.")
        if runs:
            r = runs[0]
            print(f"  Пример params:  {dict(list(r.data.params.items())[:6])}")
            print(f"  Пример metrics: {dict(list(r.data.metrics.items())[:6])}")
        return pd.DataFrame()

    df   = pd.DataFrame(rows)
    # Один run на ряд — дублей быть не должно, но на всякий случай берём min
    best = (df.sort_values("part1_mape")
              .groupby(["category", "channel"])
              .first()
              .reset_index())

    print(f"  Извлечено {len(best)} рядов")
    print(f"  Распределение лучших моделей части 1:")
    print(df["part1_model"].value_counts().to_string())
    print(f"  Медианный MAPE (лучшая модель): {best['part1_mape'].median():.2f}%")
    return best

    if not rows:
        print("  [WARN] Не удалось извлечь метрики.")
        return pd.DataFrame()

    df   = pd.DataFrame(rows)
    best = (df.sort_values("sarima_mape")
              .groupby(["category", "channel"])
              .first()
              .reset_index())

    print(f"  Извлечено {len(best)} рядов")
    print(f"  Медианный MAPE SARIMA: {best['sarima_mape'].median():.2f}%")
    return best


def compare_with_part1(ml_metrics: pd.DataFrame) -> pd.DataFrame:
    """
    Сравнивает ML с лучшей моделью части 1 (SARIMA/SARIMAX/Naive).
    Читает метрики из MLflow эксперимента mars-forecasting.
    """
    from src.config.settings import RESULTS_DIR

    part1_best = pd.DataFrame()

    # Сначала пробуем CSV
    for csv_name in ["test_metrics_sarima.csv", "sarima_metrics.csv",
                     "metrics.csv", "results.csv"]:
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
                                    "mape":       "part1_mape",
                                    "mae":        "part1_mae",
                                    "smape":      "part1_smape",
                                    "model_type": "part1_model",
                                }))
                break

    # Если CSV нет — читаем из MLflow
    if part1_best.empty:
        print("\n  CSV не найден. Читаем из MLflow...")
        part1_best = get_best_part1_metrics_from_mlflow()

    if part1_best.empty:
        print("\n  [WARN] Метрики части 1 недоступны.")
        print("  Сохраняем только ML метрики без сравнения.")
        return ml_metrics

    # Объединяем
    model_name = ml_metrics["ml_model"].iloc[0]
    compare    = ml_metrics.merge(
        part1_best, on=["category", "channel"], how="left"
    )

    # Победитель — сравниваем ML MAPE с лучшей моделью части 1
    compare["winner"] = compare.apply(
        lambda r: model_name
        if pd.notna(r.get("part1_mape")) and r["mape"] < r["part1_mape"]
        else r.get("part1_model", "Part1"),
        axis=1
    )
    compare["delta_mape"] = (compare["mape"] - compare["part1_mape"]).round(2)

    ml_wins    = (compare["winner"] == model_name).sum()
    part1_wins = (compare["winner"] != model_name).sum()
    no_part1   = compare["part1_mape"].isna().sum()
    total      = len(compare)

    print(f"\n=== ML vs Лучшая модель части 1 ===")
    print(f"  {model_name} лучше в {ml_wins}/{total} рядах")
    print(f"  Часть 1 лучше в {part1_wins}/{total} рядах")
    if no_part1:
        print(f"  Нет данных части 1: {no_part1} рядов")
    print(f"\n  Медианный MAPE:")
    print(f"    {model_name}: {compare['mape'].median():.2f}%")
    print(f"    Часть 1:      {compare['part1_mape'].median():.2f}%")
    print(f"    Δ медиана:    {compare['delta_mape'].median():+.2f}%")

    # Распределение победителей
    print(f"\n  Победители по рядам:")
    print(compare["winner"].value_counts().to_string())

    return compare


# =============================================================================
# ШАГ 5: ЛОГИРОВАНИЕ В MLFLOW
# =============================================================================

# Название эксперимента для сравнения моделей — используется Streamlit
MLFLOW_EXPERIMENT_COMPARISON = "mars_model_selection"


def log_per_series_to_mlflow(ml_metrics: pd.DataFrame,
                               comparison: pd.DataFrame,
                               best_info: dict):
    """
    Логирует результаты сравнения по каждому ряду в MLflow.

    Структура (для Streamlit):
      Эксперимент: mars_model_selection
        ├── Run: {category}|{channel}   ← один run на ряд
        │     ├── tags:
        │     │     category, channel,
        │     │     winner (название лучшей модели),
        │     │     winner_type (ML / Part1),
        │     │     ml_model_name,
        │     │     ml_pkl_path (путь к pkl файлу)
        │     ├── metrics:
        │     │     ml_mape, ml_mae, ml_smape,
        │     │     part1_mape, part1_mae (если есть),
        │     │     delta_mape (ML - Part1)
        │     └── params:
        │           horizon_type, evaluation
        │
        └── Run: _summary   ← сводный run
              ├── metrics: ml_wins, part1_wins, median_ml_mape, ...
              └── artifacts: ml_vs_part1_comparison.csv

    Streamlit запрашивает runs по тегам category + channel
    и читает тег winner + ml_pkl_path для загрузки модели.
    """
    import re

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_COMPARISON)

    model_name = best_info["model_name"]
    has_part1  = "part1_mape" in comparison.columns
    key_model  = "lgbm" if model_name == "LightGBM" else "xgb"

    # Пути к pkl файлам (полные — для загрузки в Streamlit)
    eval_pkl_path = str(ML_MODELS_DIR / f"ml_{key_model}_eval.pkl")
    full_pkl_path = str(ML_MODELS_DIR / f"ml_{key_model}_full.pkl")

    print(f"\n  Логируем {len(comparison)} рядов в MLflow...")
    logged = 0

    for _, row in comparison.iterrows():
        cat     = str(row["category"])
        ch      = str(row["channel"])
        ml_mape = float(row["mape"])

        # Определяем победителя
        winner      = str(row.get("winner", model_name))
        winner_type = "ML" if winner == model_name else "Part1"

        run_name = f"{cat}|{ch}"

        with mlflow.start_run(run_name=run_name):

            # Теги — главное для Streamlit
            mlflow.set_tag("category",       cat)
            mlflow.set_tag("channel",        ch)
            mlflow.set_tag("winner",         winner)
            mlflow.set_tag("winner_type",    winner_type)
            mlflow.set_tag("ml_model_name",  model_name)
            mlflow.set_tag("ml_pkl_full",    full_pkl_path)
            mlflow.set_tag("ml_pkl_eval",    eval_pkl_path)
            mlflow.set_tag("horizon_type",   "h1_12_from_T")

            # ML метрики
            ml_metrics_dict = {
                "ml_mape": round(ml_mape, 3),
            }
            if "mae" in row and pd.notna(row["mae"]):
                ml_metrics_dict["ml_mae"] = round(float(row["mae"]), 0)
            if "smape" in row and pd.notna(row.get("smape")):
                ml_metrics_dict["ml_smape"] = round(float(row["smape"]), 3)

            # Part1 метрики
            if has_part1 and pd.notna(row.get("part1_mape")):
                ml_metrics_dict["part1_mape"] = round(
                    float(row["part1_mape"]), 3
                )
                ml_metrics_dict["delta_mape"] = round(
                    ml_mape - float(row["part1_mape"]), 3
                )
                if pd.notna(row.get("part1_mae")):
                    ml_metrics_dict["part1_mae"] = round(
                        float(row["part1_mae"]), 0
                    )

            mlflow.log_metrics(ml_metrics_dict)

            # Параметры
            mlflow.log_params({
                "evaluation":    "h1_12_from_T",
                "part1_model":   str(row.get("part1_model", "unknown")),
            })

        logged += 1

    # Сводный run
    with mlflow.start_run(run_name="_summary"):
        mlflow.set_tag("run_type", "summary")
        mlflow.set_tag("ml_model", model_name)

        mlflow.log_metrics({
            "total_series":    len(comparison),
            "ml_median_mape":  round(float(comparison["mape"].median()), 3),
            "ml_mean_mape":    round(float(comparison["mape"].mean()),   3),
        })

        if has_part1:
            ml_wins    = int((comparison["winner"] == model_name).sum())
            part1_wins = int((comparison["winner"] != model_name).sum())
            mlflow.log_metrics({
                "ml_wins":            ml_wins,
                "part1_wins":         part1_wins,
                "part1_median_mape":  round(
                    float(comparison["part1_mape"].median()), 3
                ),
                "median_delta_mape":  round(
                    float(comparison["delta_mape"].median()), 3
                ),
            })
            mlflow.set_tag(
                "result",
                f"ML лучше в {ml_wins}/{len(comparison)} рядах"
            )

        # Артефакты
        comp_path = ML_RESULTS_DIR / "ml_vs_part1_comparison.csv"
        ml_path   = ML_RESULTS_DIR / "best_model_per_series.csv"
        if comp_path.exists():
            mlflow.log_artifact(str(comp_path))
        if ml_path.exists():
            mlflow.log_artifact(str(ml_path))

    print(f"  Залогировано рядов: {logged}/{len(comparison)}")
    print(f"  Эксперимент: {MLFLOW_EXPERIMENT_COMPARISON}")
    if has_part1:
        ml_wins = int((comparison["winner"] == model_name).sum())
        print(f"  ML лучше в {ml_wins}/{len(comparison)} рядах")
    print(f"\n  Streamlit использует теги:")
    print(f"    category, channel → winner, ml_pkl_full")


# =============================================================================
# ГЛАВНАЯ ФУНКЦИЯ
# =============================================================================

def main():
    print("=" * 65)
    print("ВЫБОР ЛУЧШЕЙ ML МОДЕЛИ И СРАВНЕНИЕ С SARIMA")
    print("=" * 65)

    # 1. Лучшая модель из MLflow
    best_info  = get_best_model_from_mlflow()
    model_name = best_info["model_name"]

    # 2. Загружаем данные и метаданные
    print("\nЗагружаем данные и метаданные...")
    meta_path = ML_MODELS_DIR / "ml_metadata.json"
    with open(meta_path) as f:
        meta = json.load(f)

    final_features = meta["final_features"]
    lag_ranges     = meta["ccf_lag_ranges"]

    df_raw      = load_data(str(ML_DATA_FILE))
    series_dict = create_series_dict(df_raw)
    global_df, _, _, _ = build_global_dataset(series_dict, lag_ranges)

    # 3. Загружаем eval модель (не видела тест → честная оценка)
    key       = "lgbm" if model_name == "LightGBM" else "xgb"
    eval_path = ML_MODELS_DIR / f"ml_{key}_eval.pkl"
    with open(eval_path, "rb") as f:
        trained_eval = pickle.load(f)
    print(f"  Eval модель загружена: {eval_path.name}")

    # 4. Метрики по каждому ряду
    print(f"\nСчитаем метрики по каждому ряду (h=1..12 из одной точки)...")
    ml_metrics = evaluate_per_series(
        fc=trained_eval,
        model_name=model_name,
        trained_eval=trained_eval,
        series_dict=series_dict,
        global_df=global_df,
        final_features=final_features,
    )

    print(f"\n=== МЕТРИКИ {model_name} ПО РЯДАМ ===")
    print(f"  Медиана MAPE: {ml_metrics['mape'].median():.2f}%")
    print(f"  Среднее MAPE: {ml_metrics['mape'].mean():.2f}%")
    print(f"  Мин MAPE:     {ml_metrics['mape'].min():.2f}%")
    print(f"  Макс MAPE:    {ml_metrics['mape'].max():.2f}%")
    print(f"\n  Топ-5 лучших рядов:")
    print(ml_metrics.nsmallest(5, "mape")
          [["category", "channel", "mape", "mae"]].to_string(index=False))
    print(f"\n  Топ-5 худших рядов:")
    print(ml_metrics.nlargest(5, "mape")
          [["category", "channel", "mape", "mae"]].to_string(index=False))

    # 5. Сохраняем ML метрики
    ml_path = ML_RESULTS_DIR / "best_model_per_series.csv"
    ml_metrics.to_csv(ml_path, index=False)
    print(f"\nML метрики → {ml_path}")

    # 6. Сравниваем с лучшей моделью части 1
    comparison = compare_with_part1(ml_metrics)
    comp_path  = ML_RESULTS_DIR / "ml_vs_part1_comparison.csv"
    comparison.to_csv(comp_path, index=False)
    print(f"Сравнение   → {comp_path}")

    # 7. Логируем в MLflow
    print("\nЛогируем в MLflow...")
    log_per_series_to_mlflow(ml_metrics, comparison, best_info)

    print("\n" + "=" * 65)
    print("ГОТОВО")
    print(f"MLflow: mlflow ui --backend-store-uri {MLFLOW_TRACKING_URI}")
    print("=" * 65)

    return ml_metrics, comparison


if __name__ == "__main__":
    main()