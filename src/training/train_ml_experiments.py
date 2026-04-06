"""
Эксперименты ML моделей с логированием в MLflow.
Часть 2 диплома: LightGBM и XGBoost.

Запуск:
    python -m src.training.train_ml_experiments

Предварительно должен быть запущен train_ml.py (или переданы артефакты).

Эксперименты (2-8):
  2. Кластеризация рядов по STL-характеристикам
  3. Cross-series признаки (средние по каналу/категории)
  4. Разные гиперпараметры для коротких/длинных горизонтов
  5. Ablation study — важность групп признаков
  6. Direct vs Recursive стратегия
  7. Глобальная vs Локальные модели
  8. Стекинг (Ridge мета-модель)

Все эксперименты оцениваются на одной метрике:
  CV MAPE среднее по h=1,3,6 на 3 фолдах.
  Исключение: 7 (Global vs Local) и 8 (Stacking) — на тесте,
  т.к. требуют полного train.

MLflow структура:
  mars_ml_experiments/
    ├── exp2_clustering_lgbm
    ├── exp2_clustering_xgb
    ├── exp3_cross_series_lgbm
    ...
"""

import warnings
warnings.filterwarnings("ignore")

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import lightgbm as lgb
import xgboost as xgb_module

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

from src.config.settings import (
    TARGET_COL, RANDOM_STATE, TEST_SIZE, N_FOLDS,
    SEASONAL_PERIOD, HORIZON, CV_HORIZONS,
    MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_EXP,
    ML_RESULTS_DIR,
)
from src.forecasting.ml_model import (
    DirectForecaster, cv_mape_score,
)
from src.utils.metrics import compute_metrics


# =============================================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# =============================================================================

def _train_test_split(global_df: pd.DataFrame) -> tuple:
    dates = sorted(global_df["_date"].unique())
    return (
        global_df[global_df["_date"].isin(dates[:-TEST_SIZE])].copy(),
        global_df[global_df["_date"].isin(dates[-TEST_SIZE:])].copy(),
    )


def _log_experiment(exp_name: str, model_name: str, params: dict,
                     metrics: dict, tags: dict = None):
    """Логирует один прогон эксперимента в MLflow."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_EXP)
    with mlflow.start_run(run_name=f"{exp_name}_{model_name.lower()}"):
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.set_tag("experiment", exp_name)
        mlflow.set_tag("model_type", model_name)
        mlflow.set_tag("cv_horizons", str(CV_HORIZONS))
        if tags:
            for k, v in tags.items():
                mlflow.set_tag(k, str(v))


# =============================================================================
# ЭКСПЕРИМЕНТ 2 — КЛАСТЕРИЗАЦИЯ РЯДОВ
# =============================================================================

def experiment_clustering(series_dict: dict, global_df: pd.DataFrame,
                            final_features: list, cv_folds: list,
                            make_lgbm_fn, make_xgb_fn,
                            n_clusters: int = 3) -> dict:
    """
    Разбивает ряды на кластеры по STL-характеристикам.
    Обучает отдельную модель на каждый кластер.
    Оценивается на тесте (h=1,3,6 среднее).
    """
    from statsmodels.tsa.seasonal import STL
    print("=" * 60)
    print(f"ЭКСПЕРИМЕНТ 2: Кластеризация рядов (k={n_clusters})")
    print("=" * 60)

    gdf = global_df.copy()
    train_cutoff = sorted(gdf["_date"].unique())[-TEST_SIZE]

    cluster_rows = []
    for (cat, ch), df_ in series_dict.items():
        s = df_[TARGET_COL].dropna()
        s_train = s[s.index < train_cutoff]
        if len(s_train) < 2 * SEASONAL_PERIOD:
            continue
        res    = STL(s_train, period=SEASONAL_PERIOD, robust=True).fit()
        var_r  = res.resid.var()
        var_sr = (res.seasonal + res.resid).var()
        var_tr = (res.trend + res.resid).var()
        Fs = max(0.0, 1 - var_r / var_sr) if var_sr > 0 else 0.0
        Ft = max(0.0, 1 - var_r / var_tr) if var_tr > 0 else 0.0
        cluster_rows.append({
            "category": cat, "channel": ch, "Fs": Fs, "Ft": Ft,
            "cv": s_train.std() / s_train.mean() if s_train.mean() > 0 else 0,
            "trend_slope": float(
                (res.trend.iloc[-1] - res.trend.iloc[0]) / len(res.trend)
            ),
            "seas_amplitude": float(res.seasonal.max() - res.seasonal.min()),
            "log_mean": float(np.log1p(s_train.mean())),
        })

    cluster_df = pd.DataFrame(cluster_rows)
    feat_cols  = ["Fs", "Ft", "cv", "trend_slope", "seas_amplitude", "log_mean"]
    X_cl       = StandardScaler().fit_transform(cluster_df[feat_cols].fillna(0))
    cluster_df["cluster"] = KMeans(
        n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=10
    ).fit_predict(X_cl)

    cluster_map = {(r["category"], r["channel"]): int(r["cluster"])
                   for _, r in cluster_df.iterrows()}
    gdf["_cluster"] = gdf.apply(
        lambda r: cluster_map.get((r["_category"], r["_channel"]), -1), axis=1
    )

    train_df, test_df = _train_test_split(gdf)
    results = {}

    for model_name, make_fn in [("LightGBM", make_lgbm_fn),
                                  ("XGBoost",  make_xgb_fn)]:
        rows_test = []
        cluster_models = {}

        for c in range(n_clusters):
            tr_c = train_df[train_df["_cluster"] == c]
            te_c = test_df[test_df["_cluster"] == c]
            if len(tr_c) < 50:
                continue
            X_tr = tr_c[final_features].fillna(0)
            X_te = te_c[final_features].fillna(0)

            model_c = DirectForecaster(make_fn, f"Cluster_{c}",
                                        list(range(1, HORIZON + 1)))
            model_c.fit(X_tr, tr_c)
            cluster_models[c] = model_c

            for (cat, ch), grp in te_c.groupby(["_category", "_channel"]):
                for h in CV_HORIZONS:
                    y_h   = grp[TARGET_COL].shift(-h)
                    valid = y_h.notna()
                    if valid.sum() == 0:
                        continue
                    vi = valid[valid].index
                    y_true = np.expm1(y_h.loc[vi].values)
                    y_pred = model_c.predict(X_te.loc[vi], h)
                    rows_test.append({
                        "category": cat, "channel": ch,
                        "cluster": c, "horizon": h,
                        **compute_metrics(y_true, y_pred),
                    })

        metrics_df  = pd.DataFrame(rows_test)
        clus_mape   = metrics_df["mape"].mean() if not metrics_df.empty else np.nan

        # Глобальная на тех же условиях для честного сравнения
        global_rows = []
        global_fc = DirectForecaster(make_fn, "Global", list(range(1, HORIZON + 1)))
        global_fc.fit(train_df[final_features].fillna(0), train_df)
        for (cat, ch), grp in test_df.groupby(["_category", "_channel"]):
            for h in CV_HORIZONS:
                y_h = grp[TARGET_COL].shift(-h)
                valid = y_h.notna()
                if valid.sum() == 0:
                    continue
                vi = valid[valid].index
                y_true = np.expm1(y_h.loc[vi].values)
                y_pred = global_fc.predict(
                    test_df.loc[vi][final_features].fillna(0), h
                )
                global_rows.append(compute_metrics(y_true, y_pred))
        global_mape = pd.DataFrame(global_rows)["mape"].mean()

        print(f"\n  {model_name}: кластеры MAPE={clus_mape:.2f}%  "
              f"глобальная MAPE={global_mape:.2f}%  "
              f"Δ={clus_mape - global_mape:+.2f}%")

        _log_experiment(
            "exp2_clustering", model_name,
            params={"n_clusters": n_clusters, "eval": "test h=1,3,6 mean"},
            metrics={
                "cluster_mape":  round(clus_mape, 3),
                "global_mape":   round(global_mape, 3),
                "delta_vs_global": round(clus_mape - global_mape, 3),
            },
            tags={"note": "test_based, not cv"},
        )
        results[model_name] = {
            "cluster_mape": clus_mape, "global_mape": global_mape,
        }

    return results


# =============================================================================
# ЭКСПЕРИМЕНТ 3 — CROSS-SERIES ПРИЗНАКИ
# =============================================================================

def experiment_cross_series(global_df: pd.DataFrame,
                              feature_cols: list,
                              final_features: list,
                              cv_folds: list,
                              make_lgbm_fn, make_xgb_fn) -> dict:
    """
    Добавляет среднее по каналу/категории как признаки.
    Оценивается через CV, h=1,3,6.
    """
    print("=" * 60)
    print("ЭКСПЕРИМЕНТ 3: Cross-series признаки")
    print("=" * 60)

    df = global_df.copy()

    ch_mean = df.groupby(["_channel", "_date"])[TARGET_COL].transform("mean")
    df["channel_avg_lag1"] = (df.assign(_cm=ch_mean)
                               .sort_values("_date")
                               .groupby(["_channel", "_category"])["_cm"]
                               .shift(1))
    cat_mean = df.groupby(["_category", "_date"])[TARGET_COL].transform("mean")
    df["category_avg_lag1"] = (df.assign(_cm=cat_mean)
                                .sort_values("_date")
                                .groupby(["_channel", "_category"])["_cm"]
                                .shift(1))
    df["vs_channel_lag1"] = df[TARGET_COL].shift(1) - df["channel_avg_lag1"]

    new_feats = ["channel_avg_lag1", "category_avg_lag1", "vs_channel_lag1"]
    new_final = final_features + [f for f in new_feats if f not in final_features]
    df_clean  = df.dropna(subset=new_feats).reset_index(drop=True)

    # Пересчитываем фолды для нового датасета
    dates = sorted(df_clean["_date"].unique())
    new_folds = []
    for fold in range(N_FOLDS):
        te_end   = len(dates) - fold * TEST_SIZE
        te_start = te_end - TEST_SIZE
        if te_start < 2 * SEASONAL_PERIOD:
            break
        tr_dates = dates[:te_start]
        te_dates = dates[te_start:te_end]
        new_folds.append((
            df_clean["_date"].isin(tr_dates),
            df_clean["_date"].isin(te_dates),
            fold + 1,
        ))
    new_folds.reverse()

    results = {}
    for model_name, make_fn in [("LightGBM", make_lgbm_fn),
                                  ("XGBoost",  make_xgb_fn)]:
        feats_ok_base  = [f for f in final_features if f in df_clean.columns]
        feats_ok_cross = [f for f in new_final     if f in df_clean.columns]

        base_mape  = cv_mape_score(df_clean, feats_ok_base,  new_folds,
                                    make_fn, CV_HORIZONS)
        cross_mape = cv_mape_score(df_clean, feats_ok_cross, new_folds,
                                    make_fn, CV_HORIZONS)
        delta      = cross_mape - base_mape
        verdict    = "улучшает" if delta < 0 else "не улучшает"

        print(f"\n  {model_name}: без cross={base_mape:.2f}%  "
              f"с cross={cross_mape:.2f}%  Δ={delta:+.2f}%  → {verdict}")

        _log_experiment(
            "exp3_cross_series", model_name,
            params={"new_features": str(new_feats),
                    "eval": "cv h=1,3,6 mean"},
            metrics={
                "base_cv_mape":  round(base_mape, 3),
                "cross_cv_mape": round(cross_mape, 3),
                "delta":         round(delta, 3),
            },
        )
        results[model_name] = {"base_mape": base_mape, "cross_mape": cross_mape}

    return results


# =============================================================================
# ЭКСПЕРИМЕНТ 4 — РАЗНЫЕ МОДЕЛИ ПО ГОРИЗОНТАМ
# =============================================================================

def experiment_horizon_specific(global_df: pd.DataFrame,
                                  final_features: list,
                                  cv_folds: list,
                                  make_lgbm_fn, make_xgb_fn) -> dict:
    """
    Короткие горизонты (h=1-3): сложная модель.
    Длинные (h=4-12): проще, сильнее регуляризация.
    Оценивается через CV, h=1,3,6.
    """
    print("=" * 60)
    print("ЭКСПЕРИМЕНТ 4: Разные модели по горизонтам")
    print("=" * 60)

    horizons_short = [1, 2, 3]
    horizons_long  = [4, 5, 6, 7, 8, 9, 10, 11, 12]
    eval_horizons  = CV_HORIZONS  # [1, 3, 6]

    results = {}
    for model_name, make_fn in [("LightGBM", make_lgbm_fn),
                                  ("XGBoost",  make_xgb_fn)]:
        # Для разных горизонтов используем tuned базовую модель
        # но с разными параметрами регуляризации
        if model_name == "LightGBM":
            base_params = make_lgbm_fn().get_params()
            def make_short():
                p = {**base_params, "n_estimators": 600, "reg_lambda": 0.3}
                return lgb.LGBMRegressor(**p)

            def make_long():
                p = {**base_params, "n_estimators": 300, "reg_lambda": 3.0}
                return lgb.LGBMRegressor(**p)

        else:
            base_params = make_xgb_fn().get_params()

            def make_short():
                p = {**base_params, "n_estimators": 600, "reg_lambda": 0.3}
                return xgb_module.XGBRegressor(**p)

            def make_long():
                p = {**base_params, "n_estimators": 300, "reg_lambda": 3.0}
                return xgb_module.XGBRegressor(**p)

        rows = []
        for train_mask, test_mask, fold_num in cv_folds:
            train_df = global_df.loc[train_mask].copy()
            test_df  = global_df.loc[test_mask].copy()
            X_tr     = train_df[final_features].fillna(0)
            X_te     = test_df[final_features].fillna(0)

            fc_short  = DirectForecaster(make_short, "Short", horizons_short)
            fc_long   = DirectForecaster(make_long,  "Long",  horizons_long)
            fc_single = DirectForecaster(make_fn, "Single",
                                          horizons_short + horizons_long)
            fc_short.fit(X_tr, train_df)
            fc_long.fit(X_tr,  train_df)
            fc_single.fit(X_tr, train_df)

            for h in eval_horizons:
                y_h   = test_df.groupby(
                    ["_category", "_channel"])[TARGET_COL].shift(-h)
                valid = y_h.notna()
                if valid.sum() == 0:
                    continue
                idx    = valid[valid].index
                y_true = np.expm1(y_h.loc[idx].values)

                fc_h  = fc_short if h in horizons_short else fc_long
                y_sep  = fc_h.predict(X_te.loc[idx], h)
                y_sing = fc_single.predict(X_te.loc[idx], h)
                rows.append({"fold": fold_num, "horizon": h, "strategy": "Sep",
                              **compute_metrics(y_true, y_sep)})
                rows.append({"fold": fold_num, "horizon": h, "strategy": "Single",
                              **compute_metrics(y_true, y_sing)})

        res_df   = pd.DataFrame(rows)
        sep_mape = res_df[res_df["strategy"] == "Sep"]["mape"].mean()
        sin_mape = res_df[res_df["strategy"] == "Single"]["mape"].mean()
        print(f"\n  {model_name}: раздельные={sep_mape:.2f}%  "
              f"единая={sin_mape:.2f}%  Δ={sep_mape - sin_mape:+.2f}%")

        _log_experiment(
            "exp4_horizon_specific", model_name,
            params={"short_horizons": str(horizons_short),
                    "long_horizons":  str(horizons_long),
                    "eval":           "cv h=1,3,6 mean"},
            metrics={
                "sep_cv_mape":    round(sep_mape, 3),
                "single_cv_mape": round(sin_mape, 3),
                "delta":          round(sep_mape - sin_mape, 3),
            },
        )
        results[model_name] = {"sep_mape": sep_mape, "single_mape": sin_mape}

    return results


# =============================================================================
# ЭКСПЕРИМЕНТ 5 — ABLATION STUDY
# =============================================================================

def experiment_ablation(global_df: pd.DataFrame,
                          final_features: list,
                          cv_folds: list,
                          make_lgbm_fn, make_xgb_fn) -> dict:
    """
    Последовательно убирает группы признаков.
    Оценивается через CV, h=1,3,6.
    """
    from src.config.settings import EXOG_COLS
    print("=" * 60)
    print("ЭКСПЕРИМЕНТ 5: Ablation study")
    print("=" * 60)

    groups = {
        "Все признаки (baseline)": final_features,
        "Без лагов экзогенных": [f for f in final_features
                                   if not any(
                                       col.replace(" ", "_").replace("/", "_") in f
                                       for col in EXOG_COLS)],
        "Без rolling/ewm": [f for f in final_features
                             if "rolling" not in f and "ewm" not in f],
        "Без календарных": [f for f in final_features
                             if not any(c in f for c in [
                                 "month", "quarter", "year", "is_q4", "is_summer"
                             ])],
        "Без COVID dummy": [f for f in final_features if "covid" not in f],
        "Без y_normalized": [f for f in final_features if f != "y_normalized"],
        "Только лаги y":   [f for f in final_features if f.startswith("y_lag")],
        "Только лаги y + rolling": [f for f in final_features
                                     if f.startswith("y_lag")
                                     or "rolling" in f or "ewm" in f],
    }

    results = {}
    for model_name, make_fn in [("LightGBM", make_lgbm_fn),
                                  ("XGBoost",  make_xgb_fn)]:
        model_rows = []
        baseline   = None

        for name, feats in groups.items():
            feats = [f for f in feats if f in global_df.columns]
            if not feats:
                continue
            mape = cv_mape_score(global_df, feats, cv_folds,
                                  make_fn, CV_HORIZONS)
            if name == "Все признаки (baseline)":
                baseline = mape
            delta = round(mape - baseline, 2) if baseline else 0.0
            model_rows.append({
                "Вариант": name, "N признаков": len(feats),
                "CV MAPE": round(mape, 2), "Δ": delta,
            })
            print(f"  {model_name} | {name:<40}: {mape:.2f}%  Δ={delta:+.2f}%")

        abl_df = pd.DataFrame(model_rows)
        abl_df.to_csv(
            ML_RESULTS_DIR / f"exp5_ablation_{model_name.lower()}.csv",
            index=False
        )

        _log_experiment(
            "exp5_ablation", model_name,
            params={"n_groups": len(groups), "eval": "cv h=1,3,6 mean"},
            metrics={
                "baseline_cv_mape": round(baseline, 3) if baseline else 0,
                "best_cv_mape": round(abl_df["CV MAPE"].min(), 3),
            },
            tags={"artifact": f"exp5_ablation_{model_name.lower()}.csv"},
        )
        results[model_name] = abl_df

    return results


# =============================================================================
# ЭКСПЕРИМЕНТ 6 — DIRECT vs RECURSIVE
# =============================================================================

def experiment_direct_vs_recursive(global_df: pd.DataFrame,
                                     final_features: list,
                                     series_dict: dict,
                                     cv_folds: list,
                                     make_lgbm_fn, make_xgb_fn) -> dict:
    """
    Direct:    отдельная модель для каждого h — нет накопления ошибок.
    Recursive: одна модель h=1, применяется итеративно.
    Оценивается через CV, h=1,3,6 (h=12 исключён — test_size=12).
    """
    print("=" * 60)
    print("ЭКСПЕРИМЕНТ 6: Direct vs Recursive")
    print("=" * 60)

    eval_horizons = CV_HORIZONS  # [1, 3, 6]
    results       = {}

    for model_name, make_fn in [("LightGBM", make_lgbm_fn),
                                  ("XGBoost",  make_xgb_fn)]:
        rows = []
        for train_mask, test_mask, fold_num in cv_folds:
            train_df = global_df.loc[train_mask].copy()
            test_df  = global_df.loc[test_mask].copy()
            X_train  = train_df[final_features].fillna(0)
            X_test   = test_df[final_features].fillna(0)

            fc_direct = DirectForecaster(make_fn, "Direct", eval_horizons)
            fc_direct.fit(X_train, train_df)

            y_h1_tr  = train_df.groupby(
                ["_category", "_channel"])[TARGET_COL].shift(-1)
            valid_h1 = y_h1_tr.notna()
            idx_h1   = valid_h1[valid_h1].index
            model_h1 = make_fn()
            model_h1.fit(X_train.loc[idx_h1], y_h1_tr.loc[idx_h1])

            for h in eval_horizons:
                y_h   = test_df.groupby(
                    ["_category", "_channel"])[TARGET_COL].shift(-h)
                valid = y_h.notna()
                idx   = valid[valid].index
                if len(idx) == 0:
                    continue
                y_true = np.expm1(y_h.loc[idx].values)

                y_direct = fc_direct.predict(X_test.loc[idx], h)
                rows.append({"fold": fold_num, "horizon": h,
                              "strategy": "Direct",
                              **compute_metrics(y_true, y_direct)})

                # Recursive по каждому ряду
                y_rec_all = []
                for (cat, ch), grp in test_df.loc[idx].groupby(
                    ["_category", "_channel"]
                ):
                    tr_series = train_df[
                        (train_df["_category"] == cat) &
                        (train_df["_channel"] == ch)
                    ][TARGET_COL].values
                    s_log_hist = list(tr_series)

                    tr_last = train_df[
                        (train_df["_category"] == cat) &
                        (train_df["_channel"] == ch)
                    ].index
                    if len(tr_last) == 0:
                        y_rec_all.extend([np.nan] * len(grp))
                        continue
                    x_last = X_train.loc[[tr_last[-1]]].copy()

                    grp_idx   = grp.index
                    preds_log = []
                    for step in range(1, h + 1):
                        for lag in [24, 13, 12, 6, 3, 2, 1]:
                            col = f"y_lag{lag}"
                            if col not in final_features:
                                continue
                            if lag == 1 and preds_log:
                                x_last[col] = preds_log[-1]
                            elif lag > 1:
                                src = f"y_lag{lag-1}"
                                if src in final_features:
                                    x_last[col] = x_last[src].values[0]

                        hist = np.array(s_log_hist + preds_log)
                        for w in [3, 6, 12]:
                            if f"rolling_mean_{w}" in final_features and len(hist) >= w:
                                x_last[f"rolling_mean_{w}"] = float(hist[-w:].mean())
                            if f"rolling_std_{w}" in final_features and len(hist) >= w:
                                x_last[f"rolling_std_{w}"] = float(hist[-w:].std())
                        if "ewm_03" in final_features:
                            x_last["ewm_03"] = float(
                                pd.Series(hist).ewm(alpha=0.3).mean().iloc[-1]
                            )
                        if "ewm_07" in final_features:
                            x_last["ewm_07"] = float(
                                pd.Series(hist).ewm(alpha=0.7).mean().iloc[-1]
                            )
                        if step <= len(grp_idx):
                            x_src = X_test.loc[[
                                grp_idx[min(step - 1, len(grp_idx) - 1)]
                            ]]
                            for cal in ["month", "month_sin", "month_cos",
                                         "quarter", "quarter_sin", "quarter_cos",
                                         "is_q4", "is_summer", "year"]:
                                if cal in final_features:
                                    x_last[cal] = x_src[cal].values[0]
                        y_log = float(model_h1.predict(x_last)[0])
                        preds_log.append(y_log)

                    y_rec_all.extend(
                        [float(np.expm1(preds_log[-1]))] * len(grp_idx)
                    )

                y_rec    = np.array(y_rec_all[:len(idx)])
                valid_rc = ~np.isnan(y_rec)
                if valid_rc.sum() > 0:
                    rows.append({"fold": fold_num, "horizon": h,
                                  "strategy": "Recursive",
                                  **compute_metrics(y_true[valid_rc],
                                                    y_rec[valid_rc])})

        res_df    = pd.DataFrame(rows)
        dir_mape  = res_df[res_df["strategy"] == "Direct"]["mape"].mean()
        rec_mape  = res_df[res_df["strategy"] == "Recursive"]["mape"].mean()
        print(f"\n  {model_name}: direct={dir_mape:.2f}%  "
              f"recursive={rec_mape:.2f}%  Δ={dir_mape - rec_mape:+.2f}%")

        _log_experiment(
            "exp6_direct_vs_recursive", model_name,
            params={"eval_horizons": str(eval_horizons),
                    "eval": "cv h=1,3,6 mean"},
            metrics={
                "direct_cv_mape":    round(dir_mape, 3),
                "recursive_cv_mape": round(rec_mape, 3),
                "delta_dir_vs_rec":  round(dir_mape - rec_mape, 3),
            },
        )
        results[model_name] = {"direct_mape": dir_mape, "recursive_mape": rec_mape}

    return results


# =============================================================================
# ЭКСПЕРИМЕНТ 7 — ГЛОБАЛЬНАЯ vs ЛОКАЛЬНЫЕ
# =============================================================================

def experiment_global_vs_local(global_df: pd.DataFrame,
                                 final_features: list,
                                 make_lgbm_fn, make_xgb_fn) -> dict:
    """
    Глобальная: одна модель на все ряды.
    Локальная:  отдельная модель на каждый ряд.
    Оценивается на тесте (h=1,3,6), т.к. нужен полный train.
    """
    print("=" * 60)
    print("ЭКСПЕРИМЕНТ 7: Глобальная vs Локальные")
    print("=" * 60)

    train_df, test_df = _train_test_split(global_df)
    X_train = train_df[final_features].fillna(0)
    X_test  = test_df[final_features].fillna(0)
    eval_h  = CV_HORIZONS

    results = {}
    for model_name, make_fn in [("LightGBM", make_lgbm_fn),
                                  ("XGBoost",  make_xgb_fn)]:
        global_fc = DirectForecaster(make_fn, "Global", eval_h)
        global_fc.fit(X_train, train_df)

        rows = []
        for cat, ch in test_df.groupby(["_category", "_channel"]).groups.keys():
            tr_s = train_df[
                (train_df["_category"] == cat) & (train_df["_channel"] == ch)
            ]
            te_s = test_df[
                (test_df["_category"] == cat) & (test_df["_channel"] == ch)
            ]
            if te_s.empty:
                continue
            X_te_s = te_s[final_features].fillna(0)

            y_h1_s  = tr_s.groupby(
                ["_category", "_channel"])[TARGET_COL].shift(-1)
            valid_s = y_h1_s.notna()

            if valid_s.sum() >= 15:
                X_tr_s   = tr_s[final_features].fillna(0)
                local_fc = DirectForecaster(make_fn, "Local", eval_h)
                local_fc.fit(X_tr_s, tr_s)

            for h in eval_h:
                y_h   = te_s[TARGET_COL].shift(-h)
                valid = y_h.notna()
                if valid.sum() == 0:
                    continue
                vi     = valid[valid].index
                y_true = np.expm1(y_h.loc[vi].values)

                y_global = global_fc.predict(X_test.loc[vi], h)
                rows.append({"category": cat, "channel": ch,
                              "model": "Глобальная", "horizon": h,
                              **compute_metrics(y_true, y_global)})
                if valid_s.sum() >= 15:
                    y_local = local_fc.predict(X_te_s.loc[vi], h)
                    rows.append({"category": cat, "channel": ch,
                                  "model": "Локальная", "horizon": h,
                                  **compute_metrics(y_true, y_local)})

        res_df     = pd.DataFrame(rows)
        global_m   = res_df[res_df["model"] == "Глобальная"]["mape"].mean()
        local_m    = res_df[res_df["model"] == "Локальная"]["mape"].mean()
        print(f"\n  {model_name}: глобальная={global_m:.2f}%  "
              f"локальная={local_m:.2f}%  Δ={global_m - local_m:+.2f}%")

        _log_experiment(
            "exp7_global_vs_local", model_name,
            params={"eval_horizons": str(eval_h),
                    "eval": "test h=1,3,6 mean"},
            metrics={
                "global_mape": round(global_m, 3),
                "local_mape":  round(local_m, 3),
                "delta":       round(global_m - local_m, 3),
            },
            tags={"note": "test_based_not_cv"},
        )
        results[model_name] = {"global_mape": global_m, "local_mape": local_m}

    return results


# =============================================================================
# ЭКСПЕРИМЕНТ 8 — СТЕКИНГ
# =============================================================================

def experiment_stacking(global_df: pd.DataFrame,
                          final_features: list,
                          make_lgbm_fn, make_xgb_fn) -> dict:
    """
    Мета-модель Ridge поверх OOF предсказаний LightGBM + XGBoost.
    OOF строится по временным разбивкам train — без leakage.
    Оценивается на тесте (h=1,3,6).
    """
    print("=" * 60)
    print("ЭКСПЕРИМЕНТ 8: Стекинг (LGBM + XGBoost → Ridge)")
    print("=" * 60)

    from src.forecasting.ml_model import make_elasticnet

    train_df, test_df = _train_test_split(global_df)
    X_train = train_df[final_features].fillna(0)
    X_test  = test_df[final_features].fillna(0)

    scaler  = StandardScaler()
    X_tr_sc = pd.DataFrame(scaler.fit_transform(X_train),
                            columns=final_features, index=X_train.index)
    X_te_sc = pd.DataFrame(scaler.transform(X_test),
                            columns=final_features, index=X_test.index)

    y_train_log = train_df[TARGET_COL]
    train_dates = sorted(train_df["_date"].unique())
    kfold       = 3
    fold_size   = len(train_dates) // kfold

    oof_lgbm = np.full(len(train_df), np.nan)
    oof_xgb  = np.full(len(train_df), np.nan)
    oof_en   = np.full(len(train_df), np.nan)

    for k in range(kfold):
        tr_dates  = train_dates[:k * fold_size]
        val_dates = train_dates[k * fold_size: (k + 1) * fold_size]
        if not tr_dates or not val_dates:
            continue
        tr_mask  = train_df["_date"].isin(tr_dates)
        val_mask = train_df["_date"].isin(val_dates)
        X_tr_k   = X_train.loc[tr_mask]
        X_val_k  = X_train.loc[val_mask]
        y_tr_k   = y_train_log.loc[tr_mask]
        if len(X_tr_k) < 10:
            continue

        oof_lgbm[val_mask.values] = make_lgbm_fn().fit(X_tr_k, y_tr_k).predict(X_val_k)
        oof_xgb[val_mask.values]  = make_xgb_fn().fit(X_tr_k,  y_tr_k).predict(X_val_k)
        en_k = make_elasticnet().fit(X_tr_sc.loc[tr_mask], y_tr_k)
        oof_en[val_mask.values]   = en_k.predict(X_tr_sc.loc[val_mask])

    oof_mask = ~(np.isnan(oof_lgbm) | np.isnan(oof_xgb) | np.isnan(oof_en))
    if oof_mask.sum() < 20:
        print("  [WARN] Мало OOF данных")
        return {}

    X_meta_train = np.column_stack([oof_lgbm[oof_mask],
                                     oof_xgb[oof_mask],
                                     oof_en[oof_mask]])
    y_meta_train = y_train_log.values[oof_mask]

    lgbm_f = make_lgbm_fn().fit(X_train, y_train_log)
    xgb_f  = make_xgb_fn().fit(X_train,  y_train_log)
    en_f   = make_elasticnet().fit(X_tr_sc, y_train_log)

    X_meta_test = np.column_stack([
        lgbm_f.predict(X_test),
        xgb_f.predict(X_test),
        en_f.predict(X_te_sc),
    ])

    meta = Ridge(alpha=1.0)
    meta.fit(X_meta_train, y_meta_train)
    print(f"  Веса Ridge: LGBM={meta.coef_[0]:.3f}  "
          f"XGB={meta.coef_[1]:.3f}  EN={meta.coef_[2]:.3f}")

    y_meta_all  = np.expm1(meta.predict(X_meta_test))
    y_lgbm_all  = np.expm1(lgbm_f.predict(X_test))
    y_xgb_all   = np.expm1(xgb_f.predict(X_test))

    rows = []
    for (cat, ch), grp in test_df.groupby(["_category", "_channel"]):
        for h in CV_HORIZONS:
            y_h   = grp[TARGET_COL].shift(-h)
            valid = y_h.notna()
            if valid.sum() == 0:
                continue
            vi     = valid[valid].index
            y_true = np.expm1(y_h.loc[vi].values)
            pos    = [test_df.index.get_loc(i) for i in vi]
            rows.append({"category": cat, "channel": ch, "model": "Стекинг",
                          "horizon": h,
                          **compute_metrics(y_true, y_meta_all[pos])})
            rows.append({"category": cat, "channel": ch, "model": "LightGBM",
                          "horizon": h,
                          **compute_metrics(y_true, y_lgbm_all[pos])})
            rows.append({"category": cat, "channel": ch, "model": "XGBoost",
                          "horizon": h,
                          **compute_metrics(y_true, y_xgb_all[pos])})

    stack_df  = pd.DataFrame(rows)
    stack_m   = stack_df[stack_df["model"] == "Стекинг"]["mape"].mean()
    lgbm_m    = stack_df[stack_df["model"] == "LightGBM"]["mape"].mean()
    xgb_m     = stack_df[stack_df["model"] == "XGBoost"]["mape"].mean()
    best_base = min(lgbm_m, xgb_m)
    print(f"\n  Стекинг MAPE: {stack_m:.2f}%  "
          f"LightGBM: {lgbm_m:.2f}%  XGBoost: {xgb_m:.2f}%  "
          f"Δ vs лучшая: {stack_m - best_base:+.2f}%")

    _log_experiment(
        "exp8_stacking", "LGBM+XGB+EN→Ridge",
        params={"ridge_alpha": 1.0, "kfold_oof": kfold,
                "eval": "test h=1,3,6 mean"},
        metrics={
            "stack_mape":    round(stack_m, 3),
            "lgbm_mape":     round(lgbm_m, 3),
            "xgb_mape":      round(xgb_m, 3),
            "delta_vs_best": round(stack_m - best_base, 3),
            "ridge_coef_lgbm": round(meta.coef_[0], 3),
            "ridge_coef_xgb":  round(meta.coef_[1], 3),
            "ridge_coef_en":   round(meta.coef_[2], 3),
        },
        tags={"note": "test_based_not_cv"},
    )
    return {"stack_mape": stack_m, "lgbm_mape": lgbm_m, "xgb_mape": xgb_m}


# =============================================================================
# ИТОГОВОЕ СРАВНЕНИЕ
# =============================================================================

def run_all_experiments(pipeline_artifacts: dict):
    """
    Запускает все эксперименты 2-8 для LightGBM и XGBoost.
    Строит итоговую таблицу и логирует сводку в MLflow.

    pipeline_artifacts — результат main() из train_ml.py.
    """
    global_df      = pipeline_artifacts["global_df"]
    feature_cols   = pipeline_artifacts["feature_cols"]
    final_features = pipeline_artifacts["final_features"]
    cv_folds       = pipeline_artifacts["cv_folds"]
    series_dict    = pipeline_artifacts.get("series_dict", None)
    make_lgbm_fn   = pipeline_artifacts["make_lgbm_tuned"]
    make_xgb_fn    = pipeline_artifacts["make_xgb_tuned"]

    series_dict = pipeline_artifacts.get("series_dict")
    if series_dict is None:
        from src.data.load_data import load_data, create_series_dict
        from src.config.settings import ML_DATA_FILE
        print("[INFO] series_dict не найден в артефактах — загружаем данные...")
        df_raw = load_data(str(ML_DATA_FILE))
        series_dict = create_series_dict(df_raw)

    # Базовые метрики из train_ml
    base_lgbm = cv_mape_score(global_df, final_features, cv_folds,
                               make_lgbm_fn, CV_HORIZONS)
    base_xgb  = cv_mape_score(global_df, final_features, cv_folds,
                               make_xgb_fn, CV_HORIZONS)
    print(f"\nБазовый CV MAPE: LightGBM={base_lgbm:.2f}%  XGBoost={base_xgb:.2f}%")

    all_results = {}

    print("\n\n" + "=" * 65)
    print("ЗАПУСК ЭКСПЕРИМЕНТОВ")
    print("=" * 65)

    try:
        r2 = experiment_clustering(
            series_dict, global_df, final_features, cv_folds,
            make_lgbm_fn, make_xgb_fn
        )
        all_results["clustering"] = r2
    except Exception as e:
        print(f"[WARN] Эксп.2 Кластеризация: {e}")

    try:
        r3 = experiment_cross_series(
            global_df, feature_cols, final_features, cv_folds,
            make_lgbm_fn, make_xgb_fn
        )
        all_results["cross_series"] = r3
    except Exception as e:
        print(f"[WARN] Эксп.3 Cross-series: {e}")

    try:
        r4 = experiment_horizon_specific(
            global_df, final_features, cv_folds, make_lgbm_fn, make_xgb_fn
        )
        all_results["horizon_specific"] = r4
    except Exception as e:
        print(f"[WARN] Эксп.4 Horizon-specific: {e}")

    try:
        r5 = experiment_ablation(
            global_df, final_features, cv_folds, make_lgbm_fn, make_xgb_fn
        )
        all_results["ablation"] = r5
    except Exception as e:
        print(f"[WARN] Эксп.5 Ablation: {e}")

    try:
        r6 = experiment_direct_vs_recursive(
            global_df, final_features, series_dict, cv_folds,
            make_lgbm_fn, make_xgb_fn
        )
        all_results["direct_vs_recursive"] = r6
    except Exception as e:
        print(f"[WARN] Эксп.6 Direct vs Recursive: {e}")

    try:
        r7 = experiment_global_vs_local(
            global_df, final_features, make_lgbm_fn, make_xgb_fn
        )
        all_results["global_vs_local"] = r7
    except Exception as e:
        print(f"[WARN] Эксп.7 Global vs Local: {e}")

    try:
        r8 = experiment_stacking(
            global_df, final_features, make_lgbm_fn, make_xgb_fn
        )
        all_results["stacking"] = r8
    except Exception as e:
        print(f"[WARN] Эксп.8 Стекинг: {e}")

    print("\n\n" + "=" * 65)
    print("ЭКСПЕРИМЕНТЫ ЗАВЕРШЕНЫ")
    print(f"MLflow: mlflow ui --backend-store-uri {MLFLOW_TRACKING_URI}")
    print("=" * 65)

    return all_results


# =============================================================================
# ЗАПУСК
# =============================================================================

if __name__ == "__main__":
    # Запускаем основной пайплайн
    from src.training.train_ml import main as run_pipeline
    artifacts = run_pipeline()

    # Добавляем series_dict для экспериментов которым он нужен
    #from src.data.load_data import load_data, create_series_dict
    #from src.config.settings import ML_DATA_FILE
    #df_raw      = load_data(str(ML_DATA_FILE))
    #series_dict = create_series_dict(df_raw)
    #artifacts["series_dict"] = series_dict

    # Запускаем эксперименты
    run_all_experiments(artifacts)