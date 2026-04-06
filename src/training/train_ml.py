"""
Запуск:
    python -m src.training.train_ml
"""

import warnings
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import shap
import mlflow
import mlflow.sklearn
import optuna

from sklearn.preprocessing import StandardScaler
from mapie.regression import TimeSeriesRegressor
from mapie.subsample import BlockBootstrap

from src.config.settings import (
    ML_DATA_FILE, ML_MODELS_DIR, ML_RESULTS_DIR,
    MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_ML, MLFLOW_EXPERIMENT_TUNING,
    TARGET_COL, EXOG_COLS, RANDOM_STATE,
    TEST_SIZE, N_FOLDS, HORIZON, CV_HORIZONS,
    LGBM_DEFAULT_PARAMS, XGB_DEFAULT_PARAMS,
    OPTUNA_N_TRIALS_LGBM, OPTUNA_N_TRIALS_XGB,
)
from src.data.load_data import load_data, create_series_dict
from src.data.preprocess import (
    analyze_ccf, build_ccf_lag_ranges, plot_ccf_heatmap,
    build_global_dataset, make_cv_folds,
)
from src.forecasting.ml_model import (
    DirectForecaster, ModelFactory, make_lgbm, make_xgb, make_elasticnet,
    make_lgbm_quantile, run_feature_selection, cv_mape_score,
)
from src.utils.metrics import compute_metrics, coverage_and_width

optuna.logging.set_verbosity(optuna.logging.WARNING)
np.random.seed(RANDOM_STATE)

# =============================================================================
# ШАГИ ПАЙПЛАЙНА
# =============================================================================

def step_load_data():
    """Шаг 1: Загрузка данных."""
    print("\n" + "=" * 60)
    print("ШАГ 1: ЗАГРУЗКА ДАННЫХ")
    print("=" * 60)
    df_raw      = load_data(str(ML_DATA_FILE))
    series_dict = create_series_dict(df_raw)
    return df_raw, series_dict


def step_ccf_analysis(series_dict: dict) -> dict:
    """Шаг 2: CCF анализ экзогенных переменных."""
    print("\n" + "=" * 60)
    print("ШАГ 2: CCF АНАЛИЗ")
    print("=" * 60)
    ccf_df, ccf_summary = analyze_ccf(series_dict)
    plot_ccf_heatmap(
        ccf_df,
        save_path=str(ML_RESULTS_DIR / "ccf_heatmap.png")
    )
    lag_ranges = build_ccf_lag_ranges(ccf_summary)
    print("\nДиапазоны лагов для экзогенных (на основе CCF):")
    for col, lags in lag_ranges.items():
        print(f"  {col:<35}: лаги {lags}")
    return lag_ranges


def step_build_dataset(series_dict: dict, lag_ranges: dict) -> tuple:
    """Шаг 3: Feature engineering и глобальный датасет."""
    print("\n" + "=" * 60)
    print("ШАГ 3: FEATURE ENGINEERING")
    print("=" * 60)
    global_df, feature_cols, cat_enc, ch_enc = build_global_dataset(
        series_dict, lag_ranges
    )
    return global_df, feature_cols, cat_enc, ch_enc


def step_feature_selection(global_df: pd.DataFrame,
                             feature_cols: list,
                             cv_folds: list) -> list:
    """Шаг 4: Отбор признаков."""
    print("\n" + "=" * 60)
    print("ШАГ 4: ОТБОР ПРИЗНАКОВ")
    print("=" * 60)
    return run_feature_selection(global_df, feature_cols, cv_folds)


def step_tune_hyperparams(global_df: pd.DataFrame,
                           final_features: list,
                           cv_folds: list) -> dict:
    """
    Шаг 5: Optuna тюнинг для LightGBM и XGBoost.
    Возвращает словарь с tuned параметрами.
    Логирует в MLflow эксперимент mars_ml_tuning.
    """
    print("\n" + "=" * 60)
    print("ШАГ 5: ТЮНИНГ ГИПЕРПАРАМЕТРОВ (Optuna)")
    print("=" * 60)

    import lightgbm as lgb
    import xgboost as xgb_module

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_TUNING)

    tuned_params = {}

    # ── LightGBM ──────────────────────────────────────────────────────────────
    print(f"\n[1/2] LightGBM ({OPTUNA_N_TRIALS_LGBM} trials)...")

    def lgbm_obj(trial):
        md = trial.suggest_int("max_depth", 3, 7)
        params = {
            "n_estimators":      trial.suggest_int("n_estimators", 100, 800),
            "max_depth":         md,
            "num_leaves":        trial.suggest_int(
                "num_leaves", 2, min(127, 2 ** md - 1)
            ),
            "learning_rate":     trial.suggest_float("lr", 0.01, 0.15, log=True),
            "reg_lambda":        trial.suggest_float(
                "reg_lambda", 0.1, 10.0, log=True
            ),
            "reg_alpha":         trial.suggest_float("reg_alpha", 0.0, 1.0),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "subsample":         trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree":  trial.suggest_float(
                "colsample_bytree", 0.5, 1.0
            ),
            "verbose": -1, "random_state": RANDOM_STATE,
        }
        return cv_mape_score(
            global_df, final_features, cv_folds,
            lambda: lgb.LGBMRegressor(**params),
            horizons=CV_HORIZONS,
        )

    lgbm_study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    )
    lgbm_study.optimize(lgbm_obj, n_trials=OPTUNA_N_TRIALS_LGBM,
                        show_progress_bar=True)

    lgbm_p = {**lgbm_study.best_params,
               "verbose": -1, "random_state": RANDOM_STATE}
    if "lr" in lgbm_p:
        lgbm_p["learning_rate"] = lgbm_p.pop("lr")
    tuned_params["lgbm"] = lgbm_p

    with mlflow.start_run(run_name="lgbm_optuna_tuning"):
        mlflow.log_params(lgbm_p)
        mlflow.log_metric("best_cv_mape", lgbm_study.best_value)
        mlflow.set_tag("model_type", "LightGBM")
        mlflow.set_tag("n_trials", OPTUNA_N_TRIALS_LGBM)
    print(f"  LightGBM best CV MAPE: {lgbm_study.best_value:.2f}%")

    # ── XGBoost ───────────────────────────────────────────────────────────────
    print(f"\n[2/2] XGBoost ({OPTUNA_N_TRIALS_XGB} trials)...")

    def xgb_obj(trial):
        params = {
            "n_estimators":     trial.suggest_int("n_estimators", 100, 800),
            "max_depth":        trial.suggest_int("max_depth", 3, 7),
            "learning_rate":    trial.suggest_float("lr", 0.01, 0.15, log=True),
            "reg_lambda":       trial.suggest_float(
                "reg_lambda", 0.1, 10.0, log=True
            ),
            "reg_alpha":        trial.suggest_float("reg_alpha", 0.0, 1.0),
            "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree", 0.5, 1.0
            ),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "gamma":            trial.suggest_float("gamma", 0.0, 2.0),
            "random_state": RANDOM_STATE, "verbosity": 0,
        }
        return cv_mape_score(
            global_df, final_features, cv_folds,
            lambda: xgb_module.XGBRegressor(**params),
            horizons=CV_HORIZONS,
        )

    xgb_study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    )
    xgb_study.optimize(xgb_obj, n_trials=OPTUNA_N_TRIALS_XGB,
                        show_progress_bar=True)

    xgb_p = {**xgb_study.best_params,
              "random_state": RANDOM_STATE, "verbosity": 0}
    if "lr" in xgb_p:
        xgb_p["learning_rate"] = xgb_p.pop("lr")
    tuned_params["xgb"] = xgb_p

    with mlflow.start_run(run_name="xgb_optuna_tuning"):
        mlflow.log_params(xgb_p)
        mlflow.log_metric("best_cv_mape", xgb_study.best_value)
        mlflow.set_tag("model_type", "XGBoost")
        mlflow.set_tag("n_trials", OPTUNA_N_TRIALS_XGB)
    print(f"  XGBoost best CV MAPE: {xgb_study.best_value:.2f}%")

    # Сохраняем параметры
    params_path = ML_MODELS_DIR / "tuned_params.json"
    with open(params_path, "w") as f:
        to_save = {
            k: {p: v for p, v in v.items() if p not in ("verbose", "verbosity")}
            for k, v in tuned_params.items()
        }
        json.dump(to_save, f, ensure_ascii=False, indent=2)
    print(f"\nПараметры сохранены → {params_path}")

    return tuned_params


def step_walk_forward_cv(global_df: pd.DataFrame,
                          final_features: list,
                          cv_folds: list,
                          make_lgbm_fn, make_xgb_fn) -> pd.DataFrame:
    """
    Шаг 6: Walk-forward CV оценка для LightGBM и XGBoost.
    Горизонты: h=1,3,6 (h=12 нельзя при test_size=12).
    """
    print("\n" + "=" * 60)
    print("ШАГ 6: WALK-FORWARD CV")
    print("=" * 60)

    results = []
    for train_mask, test_mask, fold_num in cv_folds:
        print(f"\n--- Фолд {fold_num} ---")
        train_df = global_df.loc[train_mask].copy()
        test_df  = global_df.loc[test_mask].copy()
        X_train  = train_df[final_features].fillna(0)
        X_test   = test_df[final_features].fillna(0)

        forecasters = {
            "LightGBM": DirectForecaster(make_lgbm_fn, "LightGBM", CV_HORIZONS),
            "XGBoost":  DirectForecaster(make_xgb_fn,  "XGBoost",  CV_HORIZONS),
        }
        for name, fc in forecasters.items():
            fc.fit(X_train, train_df)
            print(f"  {name} обучен")

        for h in CV_HORIZONS:
            y_h   = test_df.groupby(
                ["_category", "_channel"])[TARGET_COL].shift(-h)
            valid = y_h.notna()
            if valid.sum() == 0:
                continue
            idx    = valid[valid].index
            y_true = np.expm1(y_h.loc[idx].values)

            for name, fc in forecasters.items():
                y_pred = fc.predict(X_test.loc[idx], h)
                results.append({
                    "fold": fold_num, "horizon": h, "model": name,
                    **compute_metrics(y_true, y_pred),
                })

    cv_results = pd.DataFrame(results)
    print("\n=== ИТОГИ CV (среднее по фолдам) ===")
    print(cv_results.groupby(["model", "horizon"])[["mape", "mae"]]
          .mean().round(2).to_string())
    cv_results.to_csv(ML_RESULTS_DIR / "cv_results.csv", index=False)
    return cv_results


def step_train_final_models(global_df: pd.DataFrame,
                              final_features: list,
                              make_lgbm_fn, make_xgb_fn) -> dict:
    """
    Шаг 7: Финальное обучение.

    trained      — без последних TEST_SIZE месяцев → для evaluate_on_test
    trained_full — все данные                      → для прогноза на будущее
    """
    print("\n" + "=" * 60)
    print("ШАГ 7: ФИНАЛЬНОЕ ОБУЧЕНИЕ")
    print("=" * 60)

    dates       = sorted(global_df["_date"].unique())
    train_dates = dates[:-TEST_SIZE]
    test_dates  = dates[-TEST_SIZE:]

    train_df = global_df[global_df["_date"].isin(train_dates)].copy()
    test_df  = global_df[global_df["_date"].isin(test_dates)].copy()
    X_train  = train_df[final_features].fillna(0)
    X_test   = test_df[final_features].fillna(0)

    scaler  = StandardScaler()
    X_tr_sc = pd.DataFrame(scaler.fit_transform(X_train),
                            columns=final_features, index=X_train.index)
    X_te_sc = pd.DataFrame(scaler.transform(X_test),
                            columns=final_features, index=X_test.index)

    horizons = list(range(1, HORIZON + 1))

    lgbm_fc   = DirectForecaster(make_lgbm_fn, "LightGBM",  horizons)
    xgb_fc    = DirectForecaster(make_xgb_fn,  "XGBoost",   horizons)
    en_fc     = DirectForecaster(make_elasticnet, "ElasticNet", horizons)
    lgbm_q10  = DirectForecaster(
        lambda: make_lgbm_quantile(0.1), "LGBM_Q10", horizons
    )
    lgbm_q90  = DirectForecaster(
        lambda: make_lgbm_quantile(0.9), "LGBM_Q90", horizons
    )

    lgbm_fc.fit(X_train,  train_df)
    xgb_fc.fit(X_train,   train_df)
    en_fc.fit(X_tr_sc,    train_df)
    lgbm_q10.fit(X_train, train_df)
    lgbm_q90.fit(X_train, train_df)
    print("Финальные модели (train-часть) обучены.")

    # Полный датасет для прогноза
    X_full = global_df[final_features].fillna(0)
    lgbm_full   = DirectForecaster(make_lgbm_fn, "LightGBM_full", horizons)
    xgb_full    = DirectForecaster(make_xgb_fn,  "XGBoost_full",  horizons)
    lgbm_q10_f  = DirectForecaster(
        lambda: make_lgbm_quantile(0.1), "LGBM_Q10_full", horizons
    )
    lgbm_q90_f  = DirectForecaster(
        lambda: make_lgbm_quantile(0.9), "LGBM_Q90_full", horizons
    )
    lgbm_full.fit(X_full,   global_df)
    xgb_full.fit(X_full,    global_df)
    lgbm_q10_f.fit(X_full,  global_df)
    lgbm_q90_f.fit(X_full,  global_df)
    print("Модели на полном датасете обучены.")

    return {
        # Для оценки
        "lgbm": lgbm_fc, "xgb": xgb_fc, "en": en_fc,
        "lgbm_q10": lgbm_q10, "lgbm_q90": lgbm_q90,
        "X_train": X_train, "X_test": X_test,
        "X_tr_sc": X_tr_sc, "X_te_sc": X_te_sc,
        "train_df": train_df, "test_df": test_df,
        "scaler": scaler,
        # Для прогноза
        "lgbm_full": lgbm_full, "xgb_full": xgb_full,
        "lgbm_q10_full": lgbm_q10_f, "lgbm_q90_full": lgbm_q90_f,
    }


def step_evaluate_on_test(trained: dict,
                           final_features: list) -> pd.DataFrame:
    """
    Шаг 8: Оценка на тесте.
    Direct multi-step: h=1,3,6 (каждая точка теста — отдельная стартовая).
    """
    print("\n" + "=" * 60)
    print("ШАГ 8: ОЦЕНКА НА ТЕСТЕ (h=1,3,6)")
    print("=" * 60)

    test_df = trained["test_df"]
    X_test  = trained["X_test"]
    X_te_sc = trained["X_te_sc"]
    rows    = []

    for (cat, ch), grp in test_df.groupby(["_category", "_channel"]):
        idx = grp.index
        for h in CV_HORIZONS:
            y_h   = grp[TARGET_COL].shift(-h)
            valid = y_h.notna()
            if valid.sum() == 0:
                continue
            valid_idx = valid[valid].index
            y_true    = np.expm1(y_h.loc[valid_idx].values)

            for name, fc, x_te in [
                ("LightGBM",   trained["lgbm"], X_test),
                ("XGBoost",    trained["xgb"],  X_test),
                ("ElasticNet", trained["en"],   X_te_sc),
            ]:
                y_pred = fc.predict(x_te.loc[valid_idx], h)
                rows.append({
                    "category": cat, "channel": ch,
                    "model": name, "horizon": h,
                    **compute_metrics(y_true, y_pred),
                })

            # Quantile ДИ
            y_lo = np.expm1(
                trained["lgbm_q10"].predict_log(X_test.loc[valid_idx], h)
            )
            y_hi = np.expm1(
                trained["lgbm_q90"].predict_log(X_test.loc[valid_idx], h)
            )
            ci = coverage_and_width(y_true, y_lo, y_hi)
            rows.append({
                "category": cat, "channel": ch,
                "model": "LGBM_CI", "horizon": h,
                "mae": np.nan, "rmse": np.nan,
                "mape": np.nan, "smape": np.nan,
                **ci,
            })

    out = pd.DataFrame(rows)
    print("\n=== ТЕСТ: медианный MAPE по горизонтам ===")
    models = ["LightGBM", "XGBoost", "ElasticNet"]
    print(out[out["model"].isin(models)]
          .groupby(["model", "horizon"])["mape"]
          .median().unstack().round(2).to_string())

    ci_df = out[out["model"] == "LGBM_CI"]
    if not ci_df.empty:
        for h in CV_HORIZONS:
            ci_h = ci_df[ci_df["horizon"] == h]
            if not ci_h.empty:
                print(f"  LGBM ДИ 80% h={h}: "
                      f"coverage={ci_h['coverage'].mean():.3f}")

    out.to_csv(ML_RESULTS_DIR / "test_metrics_ml.csv", index=False)
    return out


def step_evaluate_h12(trained: dict, final_features: list) -> pd.DataFrame:
    """
    Шаг 9: Оценка h=1..12 из одной точки (как SARIMA).
    Стартовая точка T = последний месяц train.
    Для каждого h: model_h предсказывает из T, обновляем только календарь.
    Результат сопоставим с SARIMA оценкой.
    """
    print("\n" + "=" * 60)
    print("ШАГ 9: ОЦЕНКА h=1..12 ИЗ ОДНОЙ ТОЧКИ (как SARIMA)")
    print("=" * 60)

    test_df  = trained["test_df"]
    train_df = trained["train_df"]
    X_train  = trained["X_train"]
    rows     = []

    for model_name, fc in [("LightGBM", trained["lgbm"]),
                             ("XGBoost",  trained["xgb"])]:
        for (cat, ch), grp_te in test_df.groupby(["_category", "_channel"]):
            tr_s = train_df[
                (train_df["_category"] == cat) & (train_df["_channel"] == ch)
            ]
            if tr_s.empty:
                continue
            x_T = X_train.loc[[tr_s.index[-1]]]

            te_dates = sorted(grp_te["_date"].unique())
            y_preds  = []
            y_trues  = []

            for h, future_date in enumerate(te_dates, start=1):
                fact_rows = grp_te[grp_te["_date"] == future_date]
                if fact_rows.empty or h not in fc.models:
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
                    ("covid",       0), ("post_covid", 0),
                ]:
                    if feat in final_features:
                        x_h[feat] = val

                if "t" in final_features and "t" in x_T.columns:
                    new_t = float(x_T["t"].values[0] + h)
                    x_h["t"] = new_t
                    if "t_squared" in final_features:
                        x_h["t_squared"] = new_t ** 2

                y_preds.append(float(fc.predict(x_h, h)[0]))
                y_trues.append(
                    float(np.expm1(fact_rows[TARGET_COL].values[0]))
                )

            if not y_preds:
                continue
            m = compute_metrics(np.array(y_trues), np.array(y_preds))
            rows.append({"category": cat, "channel": ch,
                          "model": model_name, **m})

    out = pd.DataFrame(rows)
    print("\n=== h=1..12 из одной точки (медиана MAPE по рядам) ===")
    print(out.groupby("model")["mape"].agg(["median", "mean"]).round(2))
    out.to_csv(ML_RESULTS_DIR / "test_metrics_h12.csv", index=False)
    return out


def step_shap_analysis(trained: dict, final_features: list):
    """Шаг 10: SHAP анализ для LightGBM."""
    print("\n" + "=" * 60)
    print("ШАГ 10: SHAP АНАЛИЗ")
    print("=" * 60)

    X_test   = trained["X_test"][final_features]
    model_h1 = trained["lgbm"].models[1]

    explainer   = shap.TreeExplainer(model_h1)
    shap_values = shap.Explainer(model_h1)(X_test)

    plt.figure(figsize=(10, 7))
    shap.summary_plot(shap_values, X_test, plot_type="bar",
                      max_display=20, show=False)
    plt.title("SHAP Feature Importance (LightGBM, h=1)")
    plt.tight_layout()
    plt.savefig(ML_RESULTS_DIR / "shap_importance.png",
                dpi=130, bbox_inches="tight")
    plt.close()

    mean_abs = np.abs(shap_values.values).mean(axis=0)
    shap_df  = pd.DataFrame({
        "feature": final_features, "mean_|shap|": mean_abs
    }).sort_values("mean_|shap|", ascending=False)
    print("\n=== ТОП-15 признаков по SHAP ===")
    print(shap_df.head(15).to_string(index=False))
    shap_df.to_csv(ML_RESULTS_DIR / "shap_importance.csv", index=False)
    return shap_df


def step_generate_forecast(series_dict: dict,
                             global_df: pd.DataFrame,
                             trained: dict,
                             final_features: list) -> pd.DataFrame:
    """
    Шаг 11: Прогноз на будущее (direct из одной точки T).
    Использует trained_full — модели обученные на ВСЕХ данных.
    """
    print("\n" + "=" * 60)
    print("ШАГ 11: ПРОГНОЗ НА БУДУЩЕЕ")
    print("=" * 60)

    X_all         = global_df[final_features].fillna(0)
    forecast_rows = []

    for (cat, ch), df_ in series_dict.items():
        s         = df_[TARGET_COL].dropna()
        last_date = s.index[-1]
        future_idx = pd.date_range(
            last_date + pd.DateOffset(months=1),
            periods=HORIZON, freq="MS"
        )

        mask_s       = ((global_df["_category"] == cat) &
                         (global_df["_channel"]  == ch))
        last_row_idx = global_df.loc[mask_s].index[-1]
        x_T          = X_all.loc[[last_row_idx]].copy()

        for h in range(1, HORIZON + 1):
            fd  = future_idx[h - 1]
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
                ("covid",       0), ("post_covid", 0),
            ]:
                if feat in final_features:
                    x_h[feat] = val

            if "t" in final_features and "t" in x_T.columns:
                new_t = float(x_T["t"].values[0] + h)
                x_h["t"] = new_t
                if "t_squared" in final_features:
                    x_h["t_squared"] = new_t ** 2

            y_lo_v = float(
                trained["lgbm_q10_full"].predict_log(x_h, h)[0]
            )
            y_hi_v = float(
                trained["lgbm_q90_full"].predict_log(x_h, h)[0]
            )
            if y_lo_v > y_hi_v:
                y_lo_v, y_hi_v = y_hi_v, y_lo_v

            for model_name, fc in [
                ("LightGBM", trained["lgbm_full"]),
                ("XGBoost",  trained["xgb_full"]),
            ]:
                y_hat = float(fc.predict(x_h, h)[0])
                forecast_rows.append({
                    "category":   cat, "channel": ch,
                    "date":       str(fd.date()),
                    "forecast":   round(max(0.0, y_hat), 2),
                    "lower_80":   round(max(0.0, float(np.expm1(y_lo_v))), 2),
                    "upper_80":   round(max(0.0, float(np.expm1(y_hi_v))), 2),
                    "model_type": model_name,
                    "horizon":    h,
                })

    fc_df = pd.DataFrame(forecast_rows)
    fc_df.to_csv(ML_RESULTS_DIR / "future_forecast_ml.csv", index=False)
    print(f"Прогноз сохранён → {ML_RESULTS_DIR / 'future_forecast_ml.csv'}")
    return fc_df


def step_log_to_mlflow(cv_results: pd.DataFrame,
                        test_metrics: pd.DataFrame,
                        test_h12: pd.DataFrame,
                        tuned_params: dict,
                        trained: dict,
                        final_features: list,
                        shap_df: pd.DataFrame,
                        cat_enc: dict,
                        ch_enc: dict,
                        lag_ranges: dict):
    """
    Шаг 12: Логирование результатов в MLflow.

    Структура экспериментов:
      mars_ml_models/
        ├── lgbm_final   — финальный LightGBM
        └── xgb_final    — финальный XGBoost

    Лучшая модель регистрируется в Model Registry с тегом best_model=True.
    """
    print("\n" + "=" * 60)
    print("ШАГ 12: ЛОГИРОВАНИЕ В MLFLOW")
    print("=" * 60)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_ML)

    # Метаданные для воспроизведения
    meta = {
        "final_features": final_features,
        "cat_encoder":    cat_enc,
        "ch_encoder":     ch_enc,
        "horizon":        HORIZON,
        "test_size":      TEST_SIZE,
        "cv_horizons":    CV_HORIZONS,
        "exog_cols":      EXOG_COLS,
        "ccf_lag_ranges": lag_ranges,
        "log_transform":  True,
    }
    meta_path = ML_MODELS_DIR / "ml_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    best_mape   = np.inf
    best_run_id = None

    for model_name in ["LightGBM", "XGBoost"]:
        run_name = f"{model_name.lower()}_final"
        with mlflow.start_run(run_name=run_name) as run:
            # Параметры
            key = "lgbm" if model_name == "LightGBM" else "xgb"
            if tuned_params and key in tuned_params:
                mlflow.log_params(tuned_params[key])
            mlflow.log_param("n_features", len(final_features))
            mlflow.log_param("test_size",  TEST_SIZE)
            mlflow.log_param("n_cv_folds", N_FOLDS)
            mlflow.log_param("cv_horizons", str(CV_HORIZONS))

            # CV метрики
            cv_m = cv_results[cv_results["model"] == model_name]
            for h in CV_HORIZONS:
                cv_h = cv_m[cv_m["horizon"] == h]["mape"].mean()
                if not np.isnan(cv_h):
                    mlflow.log_metric(f"cv_mape_h{h}", round(cv_h, 3))
            cv_mean = cv_m["mape"].mean()
            mlflow.log_metric("cv_mape_mean_h136", round(cv_mean, 3))

            # Тест метрики h=1,3,6 (прямая оценка)
            test_m = test_metrics[test_metrics["model"] == model_name]
            for h in CV_HORIZONS:
                t_h = test_m[test_m["horizon"] == h]["mape"].median()
                if not np.isnan(t_h):
                    mlflow.log_metric(f"test_mape_h{h}", round(t_h, 3))

            # Тест h=1..12 из одной точки (сопоставимо с SARIMA)
            h12_m = test_h12[test_h12["model"] == model_name]
            if not h12_m.empty:
                h12_median = h12_m["mape"].median()
                h12_mean   = h12_m["mape"].mean()
                mlflow.log_metric("test_mape_h12_median", round(h12_median, 3))
                mlflow.log_metric("test_mape_h12_mean",   round(h12_mean, 3))

            # Теги
            mlflow.set_tag("model_type",  model_name)
            mlflow.set_tag("part",        "2_ml")
            mlflow.set_tag("ready_for_production", "true")
            mlflow.set_tag("best_model",  "false")  # обновим ниже

            # Артефакты
            mlflow.log_artifact(str(ML_RESULTS_DIR / "cv_results.csv"))
            mlflow.log_artifact(str(ML_RESULTS_DIR / "test_metrics_ml.csv"))
            mlflow.log_artifact(str(ML_RESULTS_DIR / "test_metrics_h12.csv"))
            mlflow.log_artifact(str(meta_path))
            if (ML_RESULTS_DIR / "shap_importance.png").exists():
                mlflow.log_artifact(
                    str(ML_RESULTS_DIR / "shap_importance.png")
                )

            # Сохраняем модели
            fc_key = "lgbm" if model_name == "LightGBM" else "xgb"
            model_path = ML_MODELS_DIR / f"ml_{fc_key}_eval.pkl"
            with open(model_path, "wb") as f:
                pickle.dump(trained[fc_key], f)
            mlflow.log_artifact(str(model_path))

            full_key  = f"{fc_key}_full"
            full_path = ML_MODELS_DIR / f"ml_{fc_key}_full.pkl"
            with open(full_path, "wb") as f:
                pickle.dump(trained[full_key], f)
            mlflow.log_artifact(str(full_path))

            # Следим за лучшей моделью по test h=1..12 (сопоставимо с SARIMA)
            if not h12_m.empty and h12_median < best_mape:
                best_mape   = h12_median
                best_run_id = run.info.run_id

    # Помечаем лучшую модель
    if best_run_id:
        client = mlflow.tracking.MlflowClient()
        client.set_tag(best_run_id, "best_model", "true")
        print(f"\n  Лучшая модель: run_id={best_run_id}, "
              f"test MAPE h12={best_mape:.2f}%")
        print(f"  Тег best_model=true установлен.")

    print(f"\nMLflow UI: mlflow ui --backend-store-uri {MLFLOW_TRACKING_URI}")


# =============================================================================
# ГЛАВНАЯ ФУНКЦИЯ
# =============================================================================

def main():
    print("=" * 65)
    print("MARS ML PIPELINE — ЧАСТЬ 2")
    print("=" * 65)

    # 1-3. Данные
    df_raw, series_dict = step_load_data()
    lag_ranges          = step_ccf_analysis(series_dict)
    global_df, feature_cols, cat_enc, ch_enc = step_build_dataset(
        series_dict, lag_ranges
    )
    cv_folds = make_cv_folds(global_df)

    # 4. Отбор признаков
    final_features = step_feature_selection(global_df, feature_cols, cv_folds)

    # 5. Тюнинг
    tuned_params = step_tune_hyperparams(global_df, final_features, cv_folds)

    # Tuned фабрики — используем ModelFactory (pickle-совместимый класс)
    lgbm_p = tuned_params.get("lgbm", LGBM_DEFAULT_PARAMS)
    xgb_p = tuned_params.get("xgb", XGB_DEFAULT_PARAMS)

    make_lgbm_tuned = ModelFactory("lgbm", lgbm_p)
    make_xgb_tuned = ModelFactory("xgb", xgb_p)

    # 6. CV оценка
    cv_results = step_walk_forward_cv(
        global_df, final_features, cv_folds,
        make_lgbm_tuned, make_xgb_tuned
    )

    # 7. Финальное обучение
    trained = step_train_final_models(
        global_df, final_features, make_lgbm_tuned, make_xgb_tuned
    )

    # 8-9. Оценка на тесте
    test_metrics = step_evaluate_on_test(trained, final_features)
    test_h12     = step_evaluate_h12(trained, final_features)

    # 10. SHAP
    shap_df = step_shap_analysis(trained, final_features)

    # 11. Прогноз
    future_fc = step_generate_forecast(
        series_dict, global_df, trained, final_features
    )

    # 12. MLflow
    step_log_to_mlflow(
        cv_results, test_metrics, test_h12,
        tuned_params, trained, final_features,
        shap_df, cat_enc, ch_enc, lag_ranges,
    )

    # Сохраняем метаданные
    with open(ML_MODELS_DIR / "ml_metadata.json", "w") as f:
        json.dump({
            "final_features": final_features,
            "cat_encoder":    cat_enc,
            "ch_encoder":     ch_enc,
            "horizon":        HORIZON,
            "test_size":      TEST_SIZE,
            "cv_horizons":    CV_HORIZONS,
            "exog_cols":      EXOG_COLS,
            "ccf_lag_ranges": lag_ranges,
            "lgbm_params":    lgbm_p,
            "xgb_params":     xgb_p,
        }, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 65)
    print("ПАЙПЛАЙН ЗАВЕРШЁН")
    print(f"Результаты: {ML_RESULTS_DIR}")
    print(f"Модели:     {ML_MODELS_DIR}")
    print(f"MLflow:     mlflow ui --backend-store-uri {MLFLOW_TRACKING_URI}")
    print("=" * 65)

    return {
        "global_df":     global_df, "feature_cols":  feature_cols,
        "final_features": final_features,
        "cv_folds":      cv_folds, "cv_results":    cv_results,
        "trained":       trained,  "test_metrics":  test_metrics,
        "test_h12":      test_h12, "shap_df":       shap_df,
        "future_fc":     future_fc, "cat_enc":       cat_enc,
        "ch_enc":        ch_enc, "lag_ranges":    lag_ranges,
        "tuned_params":  tuned_params,
        "make_lgbm_tuned": make_lgbm_tuned,
        "make_xgb_tuned":  make_xgb_tuned,
    }


if __name__ == "__main__":
    main()