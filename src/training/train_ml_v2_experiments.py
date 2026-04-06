"""
train_ml_v2_experiments.py — эксперименты для финальной ML v2 модели.

МЕСТО: src/training/train_ml_v2_experiments.py

ЗАПУСК:
    python -m src.training.train_ml_v2_experiments

ПРЕДУСЛОВИЯ:
    train_ml_exog_v2.py уже запускался →
        ml_v2_vif_exog_metadata.json   (метаданные финальной модели)
        ml_v2_prod_full.pkl             (зафиксированный XGBoost vif_exog)
        test_metrics_v2_vif_exog.csv   (базовые метрики)

ЭКСПЕРИМЕНТЫ:
    E2. Кластеризация рядов по STL-характеристикам
        Вопрос: улучшает ли обучение отдельных моделей на кластерах
        vs глобальная модель?

    E3. Cross-series признаки (средние по каналу/категории)
        Вопрос: помогает ли модели знать «как ведут себя соседние ряды»?

    E4. Ablation study — важность групп признаков
        Вопрос: что вносит наибольший вклад в качество?
        Группы: базовые лаги / rolling / календарь / EXOG (5 VIF-переменных) /
                 конкретные экзогенные переменные по одной.
        *** Самый ценный эксперимент — нет аналога в baseline. ***

    E5. Разные горизонты: short (h=1-3) vs long (h=4-12)
        Вопрос: стоит ли разделять модели для коротких и длинных горизонтов?

    E6. Direct vs Recursive
        Вопрос: лучше ли стратегия с отдельной моделью per-h
        чем итеративное применение h=1 модели?

    E7. Глобальная vs Локальные модели
        Вопрос: выигрывает ли индивидуальная модель на каждый ряд?

    E8. Стекинг (LightGBM + XGBoost + ElasticNet → Ridge)
        Вопрос: улучшает ли мета-ансамбль качество лучшей базовой модели?

КЛЮЧЕВОЕ ОТЛИЧИЕ ОТ baseline экспериментов:
    Все эксперименты используют DirectForecasterV2 с 5 VIF-переменными.
    Оценка: no_exog MAPE (честный сценарий без ввода пользователя).
    Базовая линия: MAPE финальной XGBoost vif_exog модели из train_ml_exog_v2.

MLflow: mars_ml_v2_experiments
"""

from __future__ import annotations

import json
import warnings
import pickle

import numpy as np
import pandas as pd
import mlflow
import lightgbm as lgb
import xgboost as xgb_module

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, ElasticNetCV
from statsmodels.tsa.seasonal import STL

from src.config.settings import (
    TARGET_COL, RANDOM_STATE, TEST_SIZE, SEASONAL_PERIOD,
    HORIZON, CV_HORIZONS, MLFLOW_TRACKING_URI,
    ML_MODELS_DIR, ML_RESULTS_DIR, ML_DATA_FILE,
    LGBM_DEFAULT_PARAMS, XGB_DEFAULT_PARAMS, EN_DEFAULT_PARAMS,
)
from src.data.load_data import load_data, create_series_dict
from src.data.preprocess import build_global_dataset, make_cv_folds
from src.utils.metrics import compute_metrics
from src.forecasting.ml_v2_model import (
    DirectForecasterV2, ModelFactoryV2,
    ARCH_NO_EXOG, ARCH_NO_FLAGS, ARCH_WITH_FLAGS,
    _safe, _has_flag, _future_col,
)

warnings.filterwarnings("ignore")
np.random.seed(RANDOM_STATE)

MLFLOW_EXPERIMENT_V2_EXP = "mars_ml_v2_experiments"
PROD_CONFIG = "vif_exog"
PROD_MODEL  = "XGBoost"


# =============================================================================
# Загрузка артефактов финальной модели
# =============================================================================

def load_production_artifacts() -> dict:
    """
    Загружает всё что нужно для экспериментов из метаданных финальной модели.
    Возвращает словарь с global_df_v2, base_features, selected_exog и т.д.
    """
    meta_path = ML_MODELS_DIR / f"ml_v2_{PROD_CONFIG}_metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"Метаданные не найдены: {meta_path}. "
            "Запустите train_ml_exog_v2.py"
        )
    with open(meta_path) as f:
        meta = json.load(f)

    print(f"  Загружаем артефакты: {PROD_MODEL} / {PROD_CONFIG}")
    print(f"  Архитектура: {meta['architecture']}")
    print(f"  Экзогенные: {meta['selected_exog']}")
    print(f"  Базовых признаков: {len(meta['base_features'])}")

    # Загружаем данные
    df_raw      = load_data(str(ML_DATA_FILE))
    series_dict = create_series_dict(df_raw)
    lag_ranges  = meta.get("ccf_lag_ranges", {})

    global_df, feature_cols, cat_enc, ch_enc = build_global_dataset(
        series_dict, lag_ranges
    )

    # Мёржим экзогенные (как в train_ml_exog_v2.step_add_future_exog)
    selected_exog = meta["selected_exog"]
    horizons      = list(range(1, HORIZON + 1))

    dates        = sorted(global_df["_date"].unique())
    train_cutoff = max(set(dates[:-TEST_SIZE]))
    df = global_df.copy()

    exog_frames = []
    for (cat, ch), raw_df in series_dict.items():
        cols_here = [c for c in selected_exog if c in raw_df.columns]
        if not cols_here:
            continue
        tmp = raw_df[cols_here].copy()
        tmp["_category"] = cat
        tmp["_channel"]  = ch
        tmp["_date"]     = tmp.index
        exog_frames.append(tmp.reset_index(drop=True))

    if exog_frames:
        exog_merged      = pd.concat(exog_frames, ignore_index=True)
        exog_merged["_date"] = pd.to_datetime(exog_merged["_date"])
        df["_date"]          = pd.to_datetime(df["_date"])
        df = df.merge(exog_merged, on=["_category", "_channel", "_date"],
                      how="left", suffixes=("", "_raw"))
        df = df.reset_index(drop=True)

    # last_value_per_series
    lv_json = meta.get("last_value_per_series", {})
    last_value_per_series = {
        tuple(k.split("|", 1)): v for k, v in lv_json.items()
    }
    global_last = meta.get("global_last", {})

    # future_h колонки
    exog_future_cols_per_h: dict[int, list] = {}
    for h in horizons:
        h_cols = []
        for col in selected_exog:
            if col not in df.columns:
                continue
            fc = _future_col(col, h)
            df[fc] = df.groupby(["_category", "_channel"])[col].shift(-h)
            h_cols.append(fc)
        exog_future_cols_per_h[h] = h_cols

    # has_flags
    for col in selected_exog:
        if col in df.columns:
            df[_has_flag(col)] = 1.0

    global_df_v2 = df
    cv_folds     = make_cv_folds(global_df_v2)

    # Tuned params
    tuned_path = ML_MODELS_DIR / f"ml_v2_{PROD_CONFIG}_tuned_params.json"
    if tuned_path.exists():
        with open(tuned_path) as f:
            tuned = json.load(f)
    else:
        tuned = {"lgbm": LGBM_DEFAULT_PARAMS, "xgb": XGB_DEFAULT_PARAMS}

    # Базовый MAPE из CSV
    base_mape = None
    csv_path = ML_RESULTS_DIR / f"test_metrics_v2_{PROD_CONFIG}.csv"
    if csv_path.exists():
        df_m = pd.read_csv(csv_path)
        sub  = df_m[(df_m["model"] == PROD_MODEL) & (df_m["mode"] == "no_exog")]
        if len(sub) > 0:
            base_mape = round(float(sub["mape"].median()), 2)
    print(f"  Базовый MAPE (no_exog, медиана): "
          f"{base_mape:.2f}%\n" if base_mape else "  Базовый MAPE: н/д\n")

    return {
        "global_df_v2":           global_df_v2,
        "feature_cols":           feature_cols,
        "base_features":          meta["base_features"],
        "selected_exog":          selected_exog,
        "exog_future_cols_per_h": exog_future_cols_per_h,
        "last_value_per_series":  last_value_per_series,
        "global_last":            global_last,
        "architecture":           meta["architecture"],
        "mask_prob":              meta.get("mask_prob", 0.3),
        "tuned_params":           tuned,
        "cv_folds":               cv_folds,
        "series_dict":            series_dict,
        "base_mape":              base_mape,
        "horizons":               horizons,
    }


# =============================================================================
# Вспомогательные
# =============================================================================

def _make_xgb(tuned: dict) -> xgb_module.XGBRegressor:
    p = {**tuned.get("xgb", XGB_DEFAULT_PARAMS),
         "verbosity": 0, "random_state": RANDOM_STATE}
    return xgb_module.XGBRegressor(**p)


def _make_lgbm(tuned: dict) -> lgb.LGBMRegressor:
    p = {**tuned.get("lgbm", LGBM_DEFAULT_PARAMS),
         "verbose": -1, "random_state": RANDOM_STATE}
    return lgb.LGBMRegressor(**p)


def _cv_mape_v2(
    global_df_v2, base_features, exog_future_cols_per_h,
    selected_exog, last_value_per_series, global_last,
    cv_folds, model_factory, architecture, mask_prob=0.3,
    eval_horizons=None,
) -> float:
    """CV MAPE в режиме no_exog для DirectForecasterV2."""
    if eval_horizons is None:
        eval_horizons = CV_HORIZONS
    mapes = []
    for train_mask, test_mask, _ in cv_folds:
        train_df = global_df_v2.loc[train_mask].copy()
        test_df  = global_df_v2.loc[test_mask].copy()

        fc = DirectForecasterV2(
            model_fn=model_factory,
            name="cv",
            horizons=eval_horizons,
            base_features=base_features,
            exog_future_cols_per_h={
                h: [c for c in cols if c in train_df.columns]
                for h, cols in exog_future_cols_per_h.items()
            },
            selected_exog=selected_exog,
            last_value_per_series=last_value_per_series,
            global_last=global_last,
            architecture=architecture,
            mask_prob=mask_prob,
        )
        fc.fit(train_df)

        for (cat, ch), grp in test_df.groupby(["_category", "_channel"]):
            tr_s = train_df[
                (train_df["_category"] == cat) & (train_df["_channel"] == ch)
            ]
            if tr_s.empty:
                continue
            x_T = tr_s[base_features].fillna(0).iloc[[-1]].copy()

            for h_idx, fd_str in enumerate(sorted(grp["_date"].unique())):
                h = h_idx + 1
                if h not in fc.models or h not in eval_horizons:
                    continue
                fact = grp[grp["_date"] == fd_str]
                if fact.empty:
                    continue
                y_true = float(np.expm1(fact[TARGET_COL].values[0]))
                fd = pd.Timestamp(fd_str)
                x_h = x_T.copy()
                for feat, val in [
                    ("month",       int(fd.month)),
                    ("month_sin",   float(np.sin(2*np.pi*fd.month/12))),
                    ("month_cos",   float(np.cos(2*np.pi*fd.month/12))),
                    ("quarter",     int(fd.quarter)),
                    ("quarter_sin", float(np.sin(2*np.pi*fd.quarter/4))),
                    ("quarter_cos", float(np.cos(2*np.pi*fd.quarter/4))),
                    ("is_q4",       int(fd.month >= 10)),
                    ("is_summer",   int(fd.month in [6, 7, 8])),
                    ("covid", 0), ("post_covid", 0),
                ]:
                    if feat in base_features:
                        x_h[feat] = val
                if "t" in base_features and "t" in x_T.columns:
                    new_t = float(x_T["t"].values[0]) + h
                    x_h["t"] = new_t
                    if "t_squared" in base_features:
                        x_h["t_squared"] = new_t ** 2

                y_pred = max(0.0, float(
                    fc.predict(x_h, h, cat=cat, ch=ch, user_exog=None)[0]
                ))
                mape_h = abs(y_true - y_pred) / (abs(y_true) + 1e-10) * 100
                if not np.isnan(mape_h):
                    mapes.append(mape_h)

    return float(np.mean(mapes)) if mapes else np.inf


def _log(run_name: str, params: dict, metrics: dict, tags: dict = None):
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_V2_EXP)
    with mlflow.start_run(run_name=run_name):
        mlflow.set_tag("prod_model",  PROD_MODEL)
        mlflow.set_tag("prod_config", PROD_CONFIG)
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        if tags:
            for k, v in tags.items():
                mlflow.set_tag(k, str(v))


def _train_test_split(df):
    dates = sorted(df["_date"].unique())
    return (
        df[df["_date"].isin(dates[:-TEST_SIZE])].copy(),
        df[df["_date"].isin(dates[-TEST_SIZE:])].copy(),
    )


def _test_mape_v2(
    global_df_v2, base_features, exog_future_cols_per_h,
    selected_exog, last_value_per_series, global_last,
    model_factory, architecture, mask_prob=0.3,
) -> float:
    """
    Тестовый MAPE в режиме no_exog — протокол идентичен step_evaluate_h12:
      T = последний месяц train, прогноз h=1..12 из одной точки.
    Это тот же протокол что даёт финальный MAPE модели (13.12%).
    Используется чтобы сравнивать результаты экспериментов с финальной моделью.
    """
    train_df, test_df = _train_test_split(global_df_v2)

    fc = DirectForecasterV2(
        model_fn=model_factory,
        name="test_eval",
        horizons=list(range(1, HORIZON + 1)),
        base_features=base_features,
        exog_future_cols_per_h={
            h: [c for c in cols if c in train_df.columns]
            for h, cols in exog_future_cols_per_h.items()
        },
        selected_exog=selected_exog,
        last_value_per_series=last_value_per_series,
        global_last=global_last,
        architecture=architecture,
        mask_prob=mask_prob,
    )
    fc.fit(train_df)

    mapes = []
    for (cat, ch), grp_te in test_df.groupby(["_category", "_channel"]):
        tr_s = train_df[(train_df["_category"] == cat) & (train_df["_channel"] == ch)]
        if tr_s.empty:
            continue
        x_T = tr_s[base_features].fillna(0).iloc[[-1]].copy()

        for h, fd_str in enumerate(sorted(grp_te["_date"].unique()), start=1):
            if h not in fc.models:
                continue
            fact = grp_te[grp_te["_date"] == fd_str]
            if fact.empty:
                continue
            y_true = float(np.expm1(fact[TARGET_COL].values[0]))
            x_h = x_T.copy()
            fd  = pd.Timestamp(fd_str)
            for feat, val in [
                ("month",       int(fd.month)),
                ("month_sin",   float(np.sin(2 * np.pi * fd.month / 12))),
                ("month_cos",   float(np.cos(2 * np.pi * fd.month / 12))),
                ("quarter",     int(fd.quarter)),
                ("quarter_sin", float(np.sin(2 * np.pi * fd.quarter / 4))),
                ("quarter_cos", float(np.cos(2 * np.pi * fd.quarter / 4))),
                ("is_q4",       int(fd.month >= 10)),
                ("is_summer",   int(fd.month in [6, 7, 8])),
                ("covid", 0), ("post_covid", 0),
            ]:
                if feat in base_features:
                    x_h[feat] = val
            if "t" in base_features and "t" in x_T.columns:
                new_t = float(x_T["t"].values[0]) + h
                x_h["t"] = new_t
                if "t_squared" in base_features:
                    x_h["t_squared"] = new_t ** 2

            y_pred = max(0.0, float(
                fc.predict(x_h, h, cat=cat, ch=ch, user_exog=None)[0]
            ))
            mape_h = abs(y_true - y_pred) / (abs(y_true) + 1e-10) * 100
            if not np.isnan(mape_h):
                mapes.append(mape_h)

    # Возвращаем медиану по рядам (как в step_evaluate_h12)
    if not mapes:
        return np.inf
    # Считаем медиану по рядам а не среднее по точкам
    row_mapes = []
    for (cat, ch), grp_te in test_df.groupby(["_category", "_channel"]):
        tr_s = train_df[(train_df["_category"] == cat) & (train_df["_channel"] == ch)]
        if tr_s.empty:
            continue
        x_T = tr_s[base_features].fillna(0).iloc[[-1]].copy()
        h_mapes = []
        for h, fd_str in enumerate(sorted(grp_te["_date"].unique()), start=1):
            if h not in fc.models:
                continue
            fact = grp_te[grp_te["_date"] == fd_str]
            if fact.empty:
                continue
            y_true = float(np.expm1(fact[TARGET_COL].values[0]))
            x_h = x_T.copy()
            fd  = pd.Timestamp(fd_str)
            for feat, val in [
                ("month", int(fd.month)),
                ("month_sin", float(np.sin(2*np.pi*fd.month/12))),
                ("month_cos", float(np.cos(2*np.pi*fd.month/12))),
                ("quarter", int(fd.quarter)),
                ("is_q4", int(fd.month >= 10)),
                ("is_summer", int(fd.month in [6, 7, 8])),
                ("covid", 0), ("post_covid", 0),
            ]:
                if feat in base_features:
                    x_h[feat] = val
            if "t" in base_features and "t" in x_T.columns:
                x_h["t"] = float(x_T["t"].values[0]) + h
            y_pred = max(0.0, float(
                fc.predict(x_h, h, cat=cat, ch=ch, user_exog=None)[0]
            ))
            m = abs(y_true - y_pred) / (abs(y_true) + 1e-10) * 100
            if not np.isnan(m):
                h_mapes.append(m)
        if h_mapes:
            row_mapes.append(float(np.mean(h_mapes)))

    return float(np.median(row_mapes)) if row_mapes else np.inf



# =============================================================================
# E2. Кластеризация рядов
# =============================================================================

def experiment_clustering(arts: dict, n_clusters: int = 3) -> dict:
    """
    Кластеризует ряды по STL-характеристикам.
    Обучает отдельный DirectForecasterV2 на каждом кластере.
    Сравнивает с глобальной моделью на тесте.

    Отличие от baseline:
      Использует DirectForecasterV2 с экзогенными в режиме no_exog.
    """
    print("=" * 60)
    print(f"E2. КЛАСТЕРИЗАЦИЯ (k={n_clusters})")
    print("=" * 60)

    global_df_v2 = arts["global_df_v2"]
    base_features = arts["base_features"]
    exog_per_h    = arts["exog_future_cols_per_h"]
    sel_exog      = arts["selected_exog"]
    lv            = arts["last_value_per_series"]
    gl            = arts["global_last"]
    arch          = arts["architecture"]
    mp            = arts["mask_prob"]
    series_dict   = arts["series_dict"]
    tuned         = arts["tuned_params"]

    # STL характеристики
    train_cutoff = sorted(global_df_v2["_date"].unique())[-TEST_SIZE]
    cluster_rows = []
    for (cat, ch), df_ in series_dict.items():
        s       = df_[TARGET_COL].dropna()
        s_train = s[s.index < train_cutoff]
        if len(s_train) < 2 * SEASONAL_PERIOD:
            continue
        try:
            res = STL(s_train, period=SEASONAL_PERIOD, robust=True).fit()
            var_r  = res.resid.var()
            var_sr = (res.seasonal + res.resid).var()
            var_tr = (res.trend + res.resid).var()
            Fs = max(0.0, 1 - var_r / var_sr) if var_sr > 0 else 0.0
            Ft = max(0.0, 1 - var_r / var_tr) if var_tr > 0 else 0.0
            cluster_rows.append({
                "category": cat, "channel": ch,
                "Fs": Fs, "Ft": Ft,
                "cv":           s_train.std() / (s_train.mean() + 1e-10),
                "trend_slope":  float((res.trend.iloc[-1] - res.trend.iloc[0]) / len(res.trend)),
                "seas_amp":     float(res.seasonal.max() - res.seasonal.min()),
                "log_mean":     float(np.log1p(s_train.mean())),
            })
        except Exception:
            continue

    cluster_df = pd.DataFrame(cluster_rows)
    feat_cols  = ["Fs", "Ft", "cv", "trend_slope", "seas_amp", "log_mean"]
    X_cl       = StandardScaler().fit_transform(cluster_df[feat_cols].fillna(0))
    cluster_df["cluster"] = KMeans(
        n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=10
    ).fit_predict(X_cl)

    cluster_map = {
        (r["category"], r["channel"]): int(r["cluster"])
        for _, r in cluster_df.iterrows()
    }

    gdf = global_df_v2.copy()
    gdf["_cluster"] = gdf.apply(
        lambda r: cluster_map.get((r["_category"], r["_channel"]), -1), axis=1
    )

    train_df, test_df = _train_test_split(gdf)

    factory_xgb = ModelFactoryV2("xgb", {
        **tuned.get("xgb", XGB_DEFAULT_PARAMS),
        "verbosity": 0, "random_state": RANDOM_STATE,
    })

    kw = dict(
        base_features=base_features,
        exog_future_cols_per_h=exog_per_h,
        selected_exog=sel_exog,
        last_value_per_series=lv,
        global_last=gl,
        architecture=arch,
        mask_prob=mp,
    )

    # Глобальная модель
    fc_global = DirectForecasterV2(
        model_fn=factory_xgb, name="global",
        horizons=CV_HORIZONS, **kw,
    )
    fc_global.fit(train_df)

    global_rows = []
    for (cat, ch), grp in test_df.groupby(["_category", "_channel"]):
        tr_s = train_df[(train_df["_category"]==cat)&(train_df["_channel"]==ch)]
        if tr_s.empty: continue
        x_T = tr_s[base_features].fillna(0).iloc[[-1]].copy()
        for h_idx, fd_str in enumerate(sorted(grp["_date"].unique())):
            h = h_idx + 1
            if h not in CV_HORIZONS or h not in fc_global.models: continue
            fact = grp[grp["_date"]==fd_str]
            if fact.empty: continue
            y_true = float(np.expm1(fact[TARGET_COL].values[0]))
            fd = pd.Timestamp(fd_str)
            x_h = x_T.copy()
            for feat, val in [
                ("month",       int(fd.month)),
                ("month_sin",   float(np.sin(2*np.pi*fd.month/12))),
                ("month_cos",   float(np.cos(2*np.pi*fd.month/12))),
                ("quarter",     int(fd.quarter)),
                ("quarter_sin", float(np.sin(2*np.pi*fd.quarter/4))),
                ("quarter_cos", float(np.cos(2*np.pi*fd.quarter/4))),
                ("is_q4",       int(fd.month >= 10)),
                ("is_summer",   int(fd.month in [6, 7, 8])),
                ("covid", 0), ("post_covid", 0),
            ]:
                if feat in base_features: x_h[feat] = val
            if "t" in base_features and "t" in x_T.columns:
                x_h["t"] = float(x_T["t"].values[0]) + h
            y_pred = max(0.0, float(fc_global.predict(x_h, h, cat=cat, ch=ch)[0]))
            global_rows.append(abs(y_true - y_pred) / (abs(y_true) + 1e-10) * 100)
    global_mape = float(np.mean(global_rows)) if global_rows else np.inf

    # Кластерные модели
    cluster_rows_test = []
    for c in range(n_clusters):
        tr_c = train_df[train_df["_cluster"] == c]
        te_c = test_df[test_df["_cluster"] == c]
        if len(tr_c) < 50: continue

        fc_c = DirectForecasterV2(
            model_fn=factory_xgb, name=f"cluster_{c}",
            horizons=CV_HORIZONS, **kw,
        )
        fc_c.fit(tr_c)

        for (cat, ch), grp in te_c.groupby(["_category", "_channel"]):
            tr_s = tr_c[(tr_c["_category"]==cat)&(tr_c["_channel"]==ch)]
            if tr_s.empty: continue
            x_T = tr_s[base_features].fillna(0).iloc[[-1]].copy()
            for h_idx, fd_str in enumerate(sorted(grp["_date"].unique())):
                h = h_idx + 1
                if h not in CV_HORIZONS or h not in fc_c.models: continue
                fact = grp[grp["_date"]==fd_str]
                if fact.empty: continue
                y_true = float(np.expm1(fact[TARGET_COL].values[0]))
                fd = pd.Timestamp(fd_str)
                x_h = x_T.copy()
                for feat, val in [
                    ("month",       int(fd.month)),
                    ("month_sin",   float(np.sin(2*np.pi*fd.month/12))),
                    ("month_cos",   float(np.cos(2*np.pi*fd.month/12))),
                    ("quarter",     int(fd.quarter)),
                    ("quarter_sin", float(np.sin(2*np.pi*fd.quarter/4))),
                    ("quarter_cos", float(np.cos(2*np.pi*fd.quarter/4))),
                    ("is_q4",       int(fd.month >= 10)),
                    ("is_summer",   int(fd.month in [6, 7, 8])),
                    ("covid", 0), ("post_covid", 0),
                ]:
                    if feat in base_features: x_h[feat] = val
                if "t" in base_features and "t" in x_T.columns:
                    x_h["t"] = float(x_T["t"].values[0]) + h
                y_pred = max(0.0, float(fc_c.predict(x_h, h, cat=cat, ch=ch)[0]))
                cluster_rows_test.append(
                    abs(y_true - y_pred) / (abs(y_true) + 1e-10) * 100
                )

    cluster_mape = float(np.mean(cluster_rows_test)) if cluster_rows_test else np.inf
    delta        = cluster_mape - global_mape
    verdict      = "улучшает" if delta < -0.1 else ("нейтрально" if abs(delta) <= 0.1 else "не улучшает")
    print(f"  Глобальная: {global_mape:.2f}%  Кластерная: {cluster_mape:.2f}%  "
          f"Δ={delta:+.2f}%  → {verdict}")

    # Состав кластеров
    for c in range(n_clusters):
        members = cluster_df[cluster_df["cluster"] == c][["category", "channel"]]
        sample  = list(members.iterrows())[:3]
        print(f"  Кластер {c} ({len(members)} рядов): "
              f"{', '.join(f'{r.category}|{r.channel}' for _, r in sample)}"
              + ("..." if len(members) > 3 else ""))

    _log(
        run_name=f"E2_clustering_k{n_clusters}",
        params={"n_clusters": n_clusters, "eval": "test no_exog h=1,3,6"},
        metrics={
            "global_mape":  round(global_mape, 3),
            "cluster_mape": round(cluster_mape, 3),
            "delta":        round(delta, 3),
        },
        tags={"verdict": verdict},
    )
    return {"global_mape": global_mape, "cluster_mape": cluster_mape}


# =============================================================================
# E3. Cross-series признаки
# =============================================================================

def experiment_cross_series(arts: dict) -> dict:
    """
    Добавляет средние по каналу/категории как дополнительные базовые признаки.
    Оценивается CV no_exog MAPE.

    Важно: cross-series признаки добавляются к base_features, не к exog.
    Экзогенные (VIF 5 переменных) остаются без изменений.
    """
    print("=" * 60)
    print("E3. CROSS-SERIES ПРИЗНАКИ")
    print("=" * 60)

    global_df_v2  = arts["global_df_v2"]
    base_features = arts["base_features"]
    exog_per_h    = arts["exog_future_cols_per_h"]
    sel_exog      = arts["selected_exog"]
    lv            = arts["last_value_per_series"]
    gl            = arts["global_last"]
    arch          = arts["architecture"]
    mp            = arts["mask_prob"]
    cv_folds      = arts["cv_folds"]
    tuned         = arts["tuned_params"]

    factory = ModelFactoryV2("xgb", {
        **tuned.get("xgb", XGB_DEFAULT_PARAMS),
        "verbosity": 0, "random_state": RANDOM_STATE,
    })

    df = global_df_v2.copy()
    # Lag-1 средних по каналу и категории (из TARGET_COL)
    ch_mean  = df.groupby(["_channel",   "_date"])[TARGET_COL].transform("mean")
    cat_mean = df.groupby(["_category",  "_date"])[TARGET_COL].transform("mean")
    df["channel_avg_lag1"]  = (df.assign(_cm=ch_mean)
                                 .groupby(["_channel", "_category"])["_cm"].shift(1))
    df["category_avg_lag1"] = (df.assign(_cm=cat_mean)
                                  .groupby(["_channel", "_category"])["_cm"].shift(1))
    df["vs_channel_lag1"]   = df[TARGET_COL].shift(1) - df["channel_avg_lag1"]

    new_feats    = ["channel_avg_lag1", "category_avg_lag1", "vs_channel_lag1"]
    cross_feats  = base_features + [f for f in new_feats if f in df.columns]

    base_mape  = _cv_mape_v2(
        global_df_v2, base_features, exog_per_h, sel_exog, lv, gl,
        cv_folds, factory, arch, mp,
    )
    cross_mape = _cv_mape_v2(
        df.dropna(subset=[f for f in new_feats if f in df.columns]).reset_index(drop=True),
        cross_feats, exog_per_h, sel_exog, lv, gl,
        cv_folds, factory, arch, mp,
    )
    delta   = cross_mape - base_mape
    verdict = "улучшает" if delta < -0.1 else ("нейтрально" if abs(delta) <= 0.1 else "не улучшает")
    print(f"  CV  — Без cross: {base_mape:.2f}%  С cross: {cross_mape:.2f}%  "
          f"Δ={delta:+.2f}%  → {verdict}")

    # Тестовая оценка (сравнима с финальной моделью h=1..12)
    df_clean     = df.dropna(subset=[f for f in new_feats if f in df.columns]).reset_index(drop=True)
    base_test    = _test_mape_v2(global_df_v2, base_features, exog_per_h,
                                  sel_exog, lv, gl, factory, arch, mp)
    cross_test   = _test_mape_v2(df_clean, cross_feats, exog_per_h,
                                  sel_exog, lv, gl, factory, arch, mp)
    delta_test   = cross_test - base_test
    print(f"  Test — Без cross: {base_test:.2f}%  С cross: {cross_test:.2f}%  "
          f"Δ={delta_test:+.2f}%")

    _log(
        run_name="E3_cross_series",
        params={"new_features": str(new_feats), "eval": "cv+test no_exog h=1,3,6/12"},
        metrics={
            "base_cv_mape":    round(base_mape, 3),
            "cross_cv_mape":   round(cross_mape, 3),
            "delta_cv":        round(delta, 3),
            "base_test_mape":  round(base_test, 3),
            "cross_test_mape": round(cross_test, 3),
            "delta_test":      round(delta_test, 3),
        },
        tags={"verdict": verdict},
    )
    return {
        "base_cv_mape": base_mape, "cross_cv_mape": cross_mape,
        "base_test_mape": base_test, "cross_test_mape": cross_test,
    }


# =============================================================================
# E4. Ablation study — важность групп признаков
# =============================================================================

def experiment_ablation(arts: dict) -> pd.DataFrame:
    """
    Последовательно убирает группы признаков и смотрит на деградацию MAPE.

    Группы для v2:
      Базовые (из baseline): лаги y / rolling+ewm / календарь / COVID dummy
      Новые для v2:
        - Все 5 VIF-экзогенных (убираем все сразу) — ключевой эксперимент
        - Каждая экзогенная по отдельности — показывает вклад каждой переменной

    *** Это самый ценный эксперимент которого нет в baseline. ***
    Показывает реальный вклад каждой из 5 VIF-переменных в качество прогноза.
    """
    print("=" * 60)
    print("E4. ABLATION STUDY (включая вклад каждой экзогенной)")
    print("=" * 60)

    global_df_v2  = arts["global_df_v2"]
    base_features = arts["base_features"]
    exog_per_h    = arts["exog_future_cols_per_h"]
    sel_exog      = arts["selected_exog"]
    lv            = arts["last_value_per_series"]
    gl            = arts["global_last"]
    arch          = arts["architecture"]
    mp            = arts["mask_prob"]
    cv_folds      = arts["cv_folds"]
    tuned         = arts["tuned_params"]

    factory = ModelFactoryV2("xgb", {
        **tuned.get("xgb", XGB_DEFAULT_PARAMS),
        "verbosity": 0, "random_state": RANDOM_STATE,
    })

    # ── Определяем группы ────────────────────────────────────────────────────
    def without(feats, fn):
        return [f for f in feats if not fn(f)]

    all_base_feats = [f for f in base_features if not f.startswith("has_")]
    groups = {}

    # Базовые группы (как в baseline)
    groups["Все признаки (baseline)"] = (
        all_base_feats, exog_per_h, sel_exog, arch
    )
    groups["Без rolling/ewm признаков"] = (
        without(all_base_feats, lambda f: "rolling" in f or "ewm" in f),
        exog_per_h, sel_exog, arch
    )
    groups["Без календарных признаков"] = (
        without(all_base_feats, lambda f: any(
            c in f for c in ["month", "quarter", "is_q4", "is_summer"]
        )),
        exog_per_h, sel_exog, arch
    )
    groups["Без COVID dummy"] = (
        without(all_base_feats, lambda f: "covid" in f),
        exog_per_h, sel_exog, arch
    )
    groups["Только лаги y + rolling"] = (
        [f for f in all_base_feats if f.startswith("y_lag") or "rolling" in f],
        exog_per_h, sel_exog, arch
    )

    # ── Экзогенные группы — ключевое для v2 ──────────────────────────────────
    # Убрать все экзогенные → ARCH_NO_EXOG
    groups["Без всех экзогенных (no_exog arch)"] = (
        without(all_base_feats, lambda f: f.startswith("has_")),
        {h: [] for h in exog_per_h},
        [],
        ARCH_NO_EXOG,
    )

    # Убрать каждую экзогенную по одной
    for col in sel_exog:
        safe_col   = _safe(col)
        remaining  = [c for c in sel_exog if c != col]
        sub_per_h  = {
            h: [fc for fc in cols if safe_col not in fc]
            for h, cols in exog_per_h.items()
        }
        base_no_col = without(all_base_feats, lambda f, s=safe_col: s in f)
        groups[f"Без {col}"] = (base_no_col, sub_per_h, remaining, arch)

    # ── Запуск ───────────────────────────────────────────────────────────────
    rows    = []
    baseline_mape      = None
    baseline_test_mape = None

    for name, (feats, ep_h, s_exog, architecture) in groups.items():
        feats_ok = [f for f in feats if f in global_df_v2.columns]
        if not feats_ok:
            print(f"  [SKIP] {name} — нет признаков")
            continue

        cv_mape   = _cv_mape_v2(
            global_df_v2, feats_ok, ep_h, s_exog, lv, gl,
            cv_folds, factory, architecture, mp,
        )
        test_mape = _test_mape_v2(
            global_df_v2, feats_ok, ep_h, s_exog, lv, gl,
            factory, architecture, mp,
        )

        if name == "Все признаки (baseline)":
            baseline_mape      = cv_mape
            baseline_test_mape = test_mape

        delta_cv   = round(cv_mape   - baseline_mape,      2) if baseline_mape      else 0.0
        delta_test = round(test_mape - baseline_test_mape, 2) if baseline_test_mape else 0.0

        tag = "★ ЭКЗОГЕННАЯ" if col in name or "экзогенных" in name else ""
        print(f"  {name:<45} CV={cv_mape:>6.2f}%  Test={test_mape:>6.2f}%  "
              f"Δcv={delta_cv:>+5.2f}%  Δtest={delta_test:>+5.2f}%  {tag}")
        rows.append({
            "Вариант":       name,
            "N признаков":   len(feats_ok) + len(s_exog),
            "CV MAPE, %":    round(cv_mape, 2),
            "Test MAPE, %":  round(test_mape, 2),
            "Δ CV":          delta_cv,
            "Δ Test":        delta_test,
            "Тип":           "exog" if "Без " in name and col in name
                             else ("no_exog" if "экзогенных" in name else "base"),
        })

    result_df = pd.DataFrame(rows)
    out_path  = ML_RESULTS_DIR / "exp_e4_ablation_v2.csv"
    result_df.to_csv(out_path, index=False)
    print(f"\n  → {out_path}")
    print(f"  Колонки: CV MAPE (3 фолда h=1,3,6) + Test MAPE (h=1..12 из одной точки T)")

    # Топ-3 важных по тестовой деградации
    exog_rows = result_df[result_df["Тип"] == "exog"].sort_values(
        "Δ Test", ascending=False
    )
    if not exog_rows.empty:
        print("\n  Топ-важных экзогенных (деградация Test MAPE при удалении):")
        for _, row in exog_rows.head(3).iterrows():
            print(f"    {row['Вариант']}: Δtest={row['Δ Test']:+.2f}%  "
                  f"Δcv={row['Δ CV']:+.2f}%")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_V2_EXP)
    with mlflow.start_run(run_name="E4_ablation"):
        mlflow.set_tag("prod_model",  PROD_MODEL)
        mlflow.set_tag("prod_config", PROD_CONFIG)
        # CV метрики
        mlflow.log_metric("baseline_cv_mape",
                          round(baseline_mape, 3) if baseline_mape else 0)
        # Test метрики — сравнимы с финальной моделью
        mlflow.log_metric("baseline_test_mape",
                          round(baseline_test_mape, 3) if baseline_test_mape else 0)
        no_exog_rows = result_df[result_df["Тип"] == "no_exog"]
        if not no_exog_rows.empty:
            mlflow.log_metric("no_exog_cv_mape",
                              round(no_exog_rows["CV MAPE, %"].values[0], 3))
            mlflow.log_metric("no_exog_test_mape",
                              round(no_exog_rows["Test MAPE, %"].values[0], 3))
        # Вклад каждой экзогенной переменной (по test MAPE)
        for _, row in exog_rows.iterrows():
            col_name = row["Вариант"].replace("Без ", "").replace(" ", "_")
            mlflow.log_metric(f"cv_impact_{col_name}",   round(row["Δ CV"],   3))
            mlflow.log_metric(f"test_impact_{col_name}", round(row["Δ Test"], 3))
        mlflow.log_artifact(str(out_path))

    return result_df


# =============================================================================
# E5. Разные горизонты: short vs long
# =============================================================================

def experiment_horizon_split(arts: dict) -> dict:
    """
    short (h=1-3): меньше регуляризации, короткий горизонт.
    long (h=4-12):  больше регуляризации, длинный горизонт.
    Оценивается CV no_exog MAPE (аналог эксп. C в train_ml_exog_v2, но подробнее).
    """
    print("=" * 60)
    print("E5. HORIZON SPLIT (short h=1-3 vs long h=4-12)")
    print("=" * 60)

    global_df_v2  = arts["global_df_v2"]
    base_features = arts["base_features"]
    exog_per_h    = arts["exog_future_cols_per_h"]
    sel_exog      = arts["selected_exog"]
    lv            = arts["last_value_per_series"]
    gl            = arts["global_last"]
    arch          = arts["architecture"]
    mp            = arts["mask_prob"]
    cv_folds      = arts["cv_folds"]
    tuned         = arts["tuned_params"]

    base_xgb = {**tuned.get("xgb", XGB_DEFAULT_PARAMS),
                "verbosity": 0, "random_state": RANDOM_STATE}
    short_hs = [h for h in range(1, HORIZON+1) if h <= 3]
    long_hs  = [h for h in range(1, HORIZON+1) if h > 3]

    # Единая модель
    unified_mape = _cv_mape_v2(
        global_df_v2, base_features, exog_per_h, sel_exog, lv, gl,
        cv_folds, ModelFactoryV2("xgb", base_xgb), arch, mp,
    )
    print(f"  Единая модель:     {unified_mape:.2f}%")

    # Split модель (разные reg_lambda)
    split_mapes = []
    for train_mask, test_mask, _ in cv_folds:
        train_df = global_df_v2.loc[train_mask].copy()
        test_df  = global_df_v2.loc[test_mask].copy()

        kw = dict(
            base_features=base_features,
            exog_future_cols_per_h={h: exog_per_h.get(h, []) for h in short_hs + long_hs},
            selected_exog=sel_exog,
            last_value_per_series=lv,
            global_last=gl,
            architecture=arch,
            mask_prob=mp,
        )
        fc_s = DirectForecasterV2(
            model_fn=ModelFactoryV2("xgb", {**base_xgb, "reg_lambda": 0.3}),
            name="short",
            horizons=short_hs,
            **{**kw, "exog_future_cols_per_h": {h: exog_per_h.get(h,[]) for h in short_hs}},
        )
        fc_l = DirectForecasterV2(
            model_fn=ModelFactoryV2("xgb", {**base_xgb, "reg_lambda": 3.0}),
            name="long",
            horizons=long_hs,
            **{**kw, "exog_future_cols_per_h": {h: exog_per_h.get(h,[]) for h in long_hs}},
        )
        fc_s.fit(train_df)
        fc_l.fit(train_df)

        for (cat, ch), grp in test_df.groupby(["_category", "_channel"]):
            tr_s = train_df[(train_df["_category"]==cat)&(train_df["_channel"]==ch)]
            if tr_s.empty: continue
            x_T = tr_s[base_features].fillna(0).iloc[[-1]].copy()
            for h_idx, fd_str in enumerate(sorted(grp["_date"].unique())):
                h = h_idx + 1
                if h not in CV_HORIZONS: continue
                fact = grp[grp["_date"]==fd_str]
                if fact.empty: continue
                y_true = float(np.expm1(fact[TARGET_COL].values[0]))
                x_h = x_T.copy()
                fd  = pd.Timestamp(fd_str)
                for feat, val in [
                    ("month", int(fd.month)), ("is_q4", int(fd.month>=10)),
                    ("is_summer", int(fd.month in [6,7,8])),
                    ("month_sin", float(np.sin(2*np.pi*fd.month/12))),
                    ("month_cos", float(np.cos(2*np.pi*fd.month/12))),
                    ("quarter", int(fd.quarter)), ("covid", 0), ("post_covid", 0),
                ]:
                    if feat in base_features: x_h[feat] = val
                fc_use = fc_s if h in short_hs else fc_l
                if h not in fc_use.models: continue
                y_pred = max(0.0, float(fc_use.predict(x_h, h, cat=cat, ch=ch)[0]))
                split_mapes.append(abs(y_true - y_pred) / (abs(y_true) + 1e-10) * 100)

    split_mape = float(np.mean(split_mapes)) if split_mapes else np.inf
    delta      = split_mape - unified_mape
    verdict    = "улучшает" if delta < -0.1 else ("нейтрально" if abs(delta) <= 0.1 else "не улучшает")
    print(f"  CV  — Unified: {unified_mape:.2f}%  Split: {split_mape:.2f}%  "
          f"Δ={delta:+.2f}%  → {verdict}")

    # Тестовая оценка (h=1..12 из одной точки T)
    unified_test = _test_mape_v2(
        global_df_v2, base_features, exog_per_h, sel_exog, lv, gl,
        ModelFactoryV2("xgb", base_xgb), arch, mp,
    )
    # Split на тесте — оцениваем финальные модели обученные на полном train
    train_df, test_df = _train_test_split(global_df_v2)
    fc_s_final = DirectForecasterV2(
        model_fn=ModelFactoryV2("xgb", {**base_xgb, "reg_lambda": 0.3}),
        name="short_final", horizons=short_hs,
        base_features=base_features,
        exog_future_cols_per_h={h: exog_per_h.get(h, []) for h in short_hs},
        selected_exog=sel_exog, last_value_per_series=lv, global_last=gl,
        architecture=arch, mask_prob=mp,
    )
    fc_l_final = DirectForecasterV2(
        model_fn=ModelFactoryV2("xgb", {**base_xgb, "reg_lambda": 3.0}),
        name="long_final", horizons=long_hs,
        base_features=base_features,
        exog_future_cols_per_h={h: exog_per_h.get(h, []) for h in long_hs},
        selected_exog=sel_exog, last_value_per_series=lv, global_last=gl,
        architecture=arch, mask_prob=mp,
    )
    fc_s_final.fit(train_df)
    fc_l_final.fit(train_df)

    split_test_errs = []
    for (cat, ch), grp in test_df.groupby(["_category", "_channel"]):
        tr_s = train_df[(train_df["_category"]==cat)&(train_df["_channel"]==ch)]
        if tr_s.empty: continue
        x_T = tr_s[base_features].fillna(0).iloc[[-1]].copy()
        for h, fd_str in enumerate(sorted(grp["_date"].unique()), start=1):
            fact = grp[grp["_date"]==fd_str]
            if fact.empty: continue
            y_true = float(np.expm1(fact[TARGET_COL].values[0]))
            x_h = x_T.copy()
            fd  = pd.Timestamp(fd_str)
            for feat, val in [
                ("month",int(fd.month)),("month_sin",float(np.sin(2*np.pi*fd.month/12))),
                ("month_cos",float(np.cos(2*np.pi*fd.month/12))),
                ("quarter",int(fd.quarter)),("is_q4",int(fd.month>=10)),
                ("is_summer",int(fd.month in[6,7,8])),("covid",0),("post_covid",0),
            ]:
                if feat in base_features: x_h[feat]=val
            fc_use = fc_s_final if h in short_hs else fc_l_final
            if h not in fc_use.models: continue
            y_pred = max(0.0, float(fc_use.predict(x_h, h, cat=cat, ch=ch)[0]))
            split_test_errs.append(abs(y_true - y_pred)/(abs(y_true)+1e-10)*100)

    split_test   = float(np.median(split_test_errs)) if split_test_errs else np.inf
    delta_test   = split_test - unified_test
    print(f"  Test — Unified: {unified_test:.2f}%  Split: {split_test:.2f}%  "
          f"Δ={delta_test:+.2f}%")

    _log(
        run_name="E5_horizon_split",
        params={"short_h": str(short_hs), "long_h": str(long_hs[:3])+"...",
                "eval": "cv+test no_exog h=1,3,6/12"},
        metrics={
            "unified_cv_mape":   round(unified_mape, 3),
            "split_cv_mape":     round(split_mape, 3),
            "delta_cv":          round(delta, 3),
            "unified_test_mape": round(unified_test, 3),
            "split_test_mape":   round(split_test, 3),
            "delta_test":        round(delta_test, 3),
        },
        tags={"verdict": verdict},
    )
    return {
        "unified_cv": unified_mape, "split_cv": split_mape,
        "unified_test": unified_test, "split_test": split_test,
    }


# =============================================================================
# E6. Direct vs Recursive
# =============================================================================

def experiment_direct_vs_recursive(arts: dict) -> dict:
    """
    Direct: отдельная DirectForecasterV2 для h=1..6.
    Recursive: модель h=1 применяется итеративно (экзогенные last_value на каждом шаге).

    Оценивается CV no_exog MAPE.
    """
    print("=" * 60)
    print("E6. DIRECT vs RECURSIVE")
    print("=" * 60)

    global_df_v2  = arts["global_df_v2"]
    base_features = arts["base_features"]
    exog_per_h    = arts["exog_future_cols_per_h"]
    sel_exog      = arts["selected_exog"]
    lv            = arts["last_value_per_series"]
    gl            = arts["global_last"]
    arch          = arts["architecture"]
    mp            = arts["mask_prob"]
    cv_folds      = arts["cv_folds"]
    tuned         = arts["tuned_params"]

    factory = ModelFactoryV2("xgb", {
        **tuned.get("xgb", XGB_DEFAULT_PARAMS),
        "verbosity": 0, "random_state": RANDOM_STATE,
    })

    direct_mape = _cv_mape_v2(
        global_df_v2, base_features, exog_per_h, sel_exog, lv, gl,
        cv_folds, factory, arch, mp, eval_horizons=CV_HORIZONS,
    )
    print(f"  Direct:    {direct_mape:.2f}%")

    # Recursive: h=1 модель итеративно
    rec_mapes = []
    for train_mask, test_mask, _ in cv_folds:
        train_df = global_df_v2.loc[train_mask].copy()
        test_df  = global_df_v2.loc[test_mask].copy()

        # Обучаем только h=1
        fc_h1 = DirectForecasterV2(
            model_fn=factory, name="h1",
            horizons=[1],
            base_features=base_features,
            exog_future_cols_per_h={1: exog_per_h.get(1, [])},
            selected_exog=sel_exog,
            last_value_per_series=lv,
            global_last=gl,
            architecture=arch,
            mask_prob=mp,
        )
        fc_h1.fit(train_df)

        for (cat, ch), grp in test_df.groupby(["_category", "_channel"]):
            tr_s = train_df[(train_df["_category"]==cat)&(train_df["_channel"]==ch)]
            if tr_s.empty: continue
            x_cur = tr_s[base_features].fillna(0).iloc[[-1]].copy()

            te_dates = sorted(grp["_date"].unique())
            for h_idx, fd_str in enumerate(te_dates):
                h = h_idx + 1
                if h not in CV_HORIZONS: continue
                fact = grp[grp["_date"]==fd_str]
                if fact.empty: continue
                y_true = float(np.expm1(fact[TARGET_COL].values[0]))

                # Итеративно применяем h=1 h раз
                x_iter = x_cur.copy()
                for step in range(h):
                    fd_step = pd.Timestamp(te_dates[min(step, len(te_dates)-1)])
                    for feat, val in [
                        ("month", int(fd_step.month)),
                        ("month_sin", float(np.sin(2*np.pi*fd_step.month/12))),
                        ("month_cos", float(np.cos(2*np.pi*fd_step.month/12))),
                        ("is_q4", int(fd_step.month>=10)),
                        ("is_summer", int(fd_step.month in [6,7,8])),
                        ("quarter", int(fd_step.quarter)),
                        ("covid", 0), ("post_covid", 0),
                    ]:
                        if feat in base_features: x_iter[feat] = val
                    if "t" in base_features:
                        x_iter["t"] = float(x_cur["t"].values[0]) + step + 1

                    # Добавляем future_h1 экзогенные в режиме no_exog
                    # (last_value per series — тот же принцип что в DirectForecasterV2)
                    for orig_col in sel_exog:
                        fc_col = _future_col(orig_col, 1)
                        if fc_col in fc_h1.feature_names_per_h.get(1, []):
                            if fc_col not in x_iter.columns:
                                x_iter[fc_col] = 0.0
                            lv_val = lv.get((cat, ch), {}).get(orig_col,
                                      gl.get(orig_col, 0.0))
                            x_iter[fc_col] = lv_val

                    # Прогнозируем h=1 — берём только признаки модели
                    pred_feats = fc_h1.feature_names_per_h.get(1, [])
                    x_pred = pd.DataFrame(0.0, index=[0], columns=pred_feats)
                    for col in pred_feats:
                        if col in x_iter.columns:
                            x_pred[col] = x_iter[col].values[0]

                    y_step_log = float(fc_h1.models[1].predict(x_pred)[0])

                    # Обновляем y_lag признаки для следующего шага
                    for lag in range(12, 0, -1):
                        if f"y_lag{lag}" in base_features and f"y_lag{lag-1}" in base_features:
                            x_iter[f"y_lag{lag}"] = x_iter[f"y_lag{lag-1}"].values[0]
                    if "y_lag1" in base_features:
                        x_iter["y_lag1"] = y_step_log

                y_pred = max(0.0, float(np.expm1(y_step_log)))
                rec_mapes.append(abs(y_true - y_pred) / (abs(y_true) + 1e-10) * 100)

    rec_mape = float(np.mean(rec_mapes)) if rec_mapes else np.inf
    delta    = direct_mape - rec_mape
    verdict  = "Direct лучше" if delta < -0.1 else ("Recursive лучше" if delta > 0.1 else "одинаково")
    print(f"  CV  — Direct: {direct_mape:.2f}%  Recursive: {rec_mape:.2f}%  "
          f"Δ(direct-rec)={delta:+.2f}%  → {verdict}")

    # Тестовая оценка direct (h=1..12 из одной точки T)
    direct_test = _test_mape_v2(
        global_df_v2, base_features, exog_per_h, sel_exog, lv, gl,
        factory, arch, mp,
    )
    print(f"  Test — Direct: {direct_test:.2f}%  "
          "(Recursive test не считается — слишком нестабилен на 12 точках)")

    _log(
        run_name="E6_direct_vs_recursive",
        params={"eval_horizons": str(CV_HORIZONS), "eval": "cv+test no_exog"},
        metrics={
            "direct_cv_mape":      round(direct_mape, 3),
            "recursive_cv_mape":   round(rec_mape, 3),
            "delta_cv_dir_vs_rec": round(delta, 3),
            "direct_test_mape":    round(direct_test, 3),
        },
        tags={"verdict": verdict},
    )
    return {
        "direct_cv": direct_mape, "recursive_cv": rec_mape,
        "direct_test": direct_test,
    }


# =============================================================================
# E7. Глобальная vs Локальные модели
# =============================================================================

def experiment_global_vs_local(arts: dict) -> dict:
    """
    Глобальная: одна DirectForecasterV2 на все 44 ряда.
    Локальная:  отдельная DirectForecasterV2 на каждый ряд.
    Оценивается на тесте (no_exog режим).
    """
    print("=" * 60)
    print("E7. ГЛОБАЛЬНАЯ vs ЛОКАЛЬНЫЕ")
    print("=" * 60)

    global_df_v2  = arts["global_df_v2"]
    base_features = arts["base_features"]
    exog_per_h    = arts["exog_future_cols_per_h"]
    sel_exog      = arts["selected_exog"]
    lv            = arts["last_value_per_series"]
    gl            = arts["global_last"]
    arch          = arts["architecture"]
    mp            = arts["mask_prob"]
    tuned         = arts["tuned_params"]

    factory = ModelFactoryV2("xgb", {
        **tuned.get("xgb", XGB_DEFAULT_PARAMS),
        "verbosity": 0, "random_state": RANDOM_STATE,
    })

    train_df, test_df = _train_test_split(global_df_v2)

    kw = dict(
        base_features=base_features,
        exog_future_cols_per_h=exog_per_h,
        selected_exog=sel_exog,
        last_value_per_series=lv,
        global_last=gl,
        architecture=arch,
        mask_prob=mp,
    )

    # Глобальная
    fc_global = DirectForecasterV2(
        model_fn=factory, name="global", horizons=CV_HORIZONS, **kw
    )
    fc_global.fit(train_df)

    global_errs, local_errs = [], []

    for (cat, ch), grp in test_df.groupby(["_category", "_channel"]):
        tr_s = train_df[(train_df["_category"]==cat)&(train_df["_channel"]==ch)]
        te_s = test_df[(test_df["_category"]==cat)&(test_df["_channel"]==ch)]
        if tr_s.empty or te_s.empty: continue

        # Локальная (нужно >= 20 точек в train)
        fc_local = None
        if len(tr_s) >= 20:
            fc_local = DirectForecasterV2(
                model_fn=factory, name=f"local_{cat}_{ch}",
                horizons=CV_HORIZONS, **kw
            )
            fc_local.fit(tr_s)

        x_T_g = tr_s[base_features].fillna(0).iloc[[-1]].copy()

        for h_idx, fd_str in enumerate(sorted(te_s["_date"].unique())):
            h = h_idx + 1
            if h not in CV_HORIZONS: continue
            fact = te_s[te_s["_date"]==fd_str]
            if fact.empty: continue
            y_true = float(np.expm1(fact[TARGET_COL].values[0]))
            x_h = x_T_g.copy()
            fd  = pd.Timestamp(fd_str)
            for feat, val in [
                ("month", int(fd.month)),
                ("month_sin", float(np.sin(2*np.pi*fd.month/12))),
                ("month_cos", float(np.cos(2*np.pi*fd.month/12))),
                ("is_q4", int(fd.month>=10)),
                ("is_summer", int(fd.month in [6,7,8])),
                ("quarter", int(fd.quarter)),
                ("covid", 0), ("post_covid", 0),
            ]:
                if feat in base_features: x_h[feat] = val

            if h in fc_global.models:
                y_glob = max(0.0, float(fc_global.predict(x_h, h, cat=cat, ch=ch)[0]))
                global_errs.append(abs(y_true - y_glob) / (abs(y_true) + 1e-10) * 100)

            if fc_local and h in fc_local.models:
                y_loc = max(0.0, float(fc_local.predict(x_h, h, cat=cat, ch=ch)[0]))
                local_errs.append(abs(y_true - y_loc) / (abs(y_true) + 1e-10) * 100)

    global_mape = float(np.mean(global_errs)) if global_errs else np.inf
    local_mape  = float(np.mean(local_errs))  if local_errs  else np.inf
    delta       = global_mape - local_mape
    verdict     = "Локальные лучше" if delta > 0.1 else ("Глобальная лучше" if delta < -0.1 else "одинаково")
    print(f"  Глобальная: {global_mape:.2f}%  Локальные: {local_mape:.2f}%  "
          f"Δ={delta:+.2f}%  → {verdict}")

    _log(
        run_name="E7_global_vs_local",
        params={"eval_horizons": str(CV_HORIZONS), "eval": "test no_exog"},
        metrics={
            "global_mape": round(global_mape, 3),
            "local_mape":  round(local_mape, 3),
            "delta":       round(delta, 3),
        },
        tags={"verdict": verdict, "note": "test_based"},
    )
    return {"global": global_mape, "local": local_mape}


# =============================================================================
# E8. Стекинг (LGBM + XGBoost + ElasticNet → Ridge)
# =============================================================================

def experiment_stacking(arts: dict) -> dict:
    """
    Мета-ансамбль поверх трёх моделей v2.
    OOF строится по временным разбивкам — без leakage.
    Все модели в режиме no_exog (честный сценарий).
    Оценивается на тесте.
    """
    print("=" * 60)
    print("E8. СТЕКИНГ (LGBM + XGBoost + ElasticNet → Ridge)")
    print("=" * 60)

    global_df_v2  = arts["global_df_v2"]
    base_features = arts["base_features"]
    exog_per_h    = arts["exog_future_cols_per_h"]
    sel_exog      = arts["selected_exog"]
    lv            = arts["last_value_per_series"]
    gl            = arts["global_last"]
    arch          = arts["architecture"]
    mp            = arts["mask_prob"]
    tuned         = arts["tuned_params"]

    train_df, test_df = _train_test_split(global_df_v2)
    train_dates = sorted(train_df["_date"].unique())
    kfold = 3

    kw = dict(
        base_features=base_features,
        exog_future_cols_per_h=exog_per_h,
        selected_exog=sel_exog,
        last_value_per_series=lv,
        global_last=gl,
        architecture=arch,
        mask_prob=mp,
        horizons=CV_HORIZONS,
    )

    factories = {
        "lgbm": (ModelFactoryV2("lgbm", {
            **tuned.get("lgbm", LGBM_DEFAULT_PARAMS),
            "verbose": -1, "random_state": RANDOM_STATE,
        }), False),
        "xgb": (ModelFactoryV2("xgb", {
            **tuned.get("xgb", XGB_DEFAULT_PARAMS),
            "verbosity": 0, "random_state": RANDOM_STATE,
        }), False),
        # ElasticNet: явные параметры для стабильного обучения.
        # scale_features=True → StandardScaler внутри DirectForecasterV2.
        # max_iter=10000 — важно: дефолт 1000 не хватает на лаговых признаках.
        # l1_ratio список — ElasticNetCV перебирает соотношение L1/L2.
        # cv=3 совпадает с kfold, не пересоздаёт лишних разбивок.
        "en": (ModelFactoryV2("elasticnet", {
            "l1_ratio":      [0.1, 0.5, 0.7, 0.9, 0.95, 1.0],
            "max_iter":      10000,
            "cv":            3,
            "fit_intercept": True,
            "random_state":  RANDOM_STATE,
        }), True),
    }

    # OOF предсказания для мета-модели
    oof = {name: {} for name in factories}  # {model_name: {(cat,ch,h): pred}}

    fold_size = len(train_dates) // kfold
    for k in range(1, kfold):
        tr_dates  = set(train_dates[:k * fold_size])
        val_dates = set(train_dates[k * fold_size: (k + 1) * fold_size])
        tr_k  = train_df[train_df["_date"].isin(tr_dates)]
        val_k = train_df[train_df["_date"].isin(val_dates)]
        if len(tr_k) < 50: continue

        for name, (factory, scale) in factories.items():
            fc = DirectForecasterV2(
                model_fn=factory, name=f"oof_{name}_k{k}",
                scale_features=scale, **kw,
            )
            fc.fit(tr_k)

            for (cat, ch), grp in val_k.groupby(["_category", "_channel"]):
                tr_s = tr_k[(tr_k["_category"]==cat)&(tr_k["_channel"]==ch)]
                if tr_s.empty: continue
                x_T = tr_s[base_features].fillna(0).iloc[[-1]].copy()
                for h_idx, fd_str in enumerate(sorted(grp["_date"].unique())):
                    h = h_idx + 1
                    if h not in CV_HORIZONS or h not in fc.models: continue
                    fact = grp[grp["_date"]==fd_str]
                    if fact.empty: continue
                    y_true = float(np.expm1(fact[TARGET_COL].values[0]))
                    x_h = x_T.copy()
                    fd  = pd.Timestamp(fd_str)
                    for feat, val in [("month",int(fd.month)),("is_q4",int(fd.month>=10)),
                                       ("covid",0),("post_covid",0)]:
                        if feat in base_features: x_h[feat]=val
                    y_pred = max(0.0, float(fc.predict(x_h, h, cat=cat, ch=ch)[0]))
                    oof[name][(cat, ch, h, fd_str)] = (y_true, y_pred)

    # Финальные модели на полном train
    final_models = {}
    for name, (factory, scale) in factories.items():
        fc = DirectForecasterV2(
            model_fn=factory, name=f"final_{name}",
            scale_features=scale, **kw,
        )
        fc.fit(train_df)
        final_models[name] = fc

    # Ridge мета-модель
    all_keys = set.intersection(*[set(oof[n].keys()) for n in factories])
    if len(all_keys) < 10:
        print("  [WARN] Мало OOF данных для стекинга")
        return {}

    y_true_oof = np.array([oof["xgb"][k][0] for k in all_keys])
    X_meta_tr  = np.column_stack([
        [oof[n][k][1] for k in all_keys] for n in factories
    ])
    meta_ridge = Ridge(alpha=1.0).fit(X_meta_tr, y_true_oof)
    print(f"  Ridge коэффициенты: "
          f"LGBM={meta_ridge.coef_[0]:.3f}  "
          f"XGB={meta_ridge.coef_[1]:.3f}  "
          f"EN={meta_ridge.coef_[2]:.3f}")

    # Оценка на тесте
    rows = {n: [] for n in list(factories.keys()) + ["stacking"]}
    for (cat, ch), grp in test_df.groupby(["_category", "_channel"]):
        tr_s = train_df[(train_df["_category"]==cat)&(train_df["_channel"]==ch)]
        if tr_s.empty: continue
        x_T = tr_s[base_features].fillna(0).iloc[[-1]].copy()

        for h_idx, fd_str in enumerate(sorted(grp["_date"].unique())):
            h = h_idx + 1
            if h not in CV_HORIZONS: continue
            fact = grp[grp["_date"]==fd_str]
            if fact.empty: continue
            y_true = float(np.expm1(fact[TARGET_COL].values[0]))
            x_h = x_T.copy()
            fd  = pd.Timestamp(fd_str)
            for feat, val in [("month",int(fd.month)),("is_q4",int(fd.month>=10)),
                               ("month_sin",float(np.sin(2*np.pi*fd.month/12))),
                               ("month_cos",float(np.cos(2*np.pi*fd.month/12))),
                               ("quarter",int(fd.quarter)),("covid",0),("post_covid",0)]:
                if feat in base_features: x_h[feat]=val

            base_preds = []
            for name in factories:
                fc = final_models[name]
                if h not in fc.models: continue
                p = max(0.0, float(fc.predict(x_h, h, cat=cat, ch=ch)[0]))
                rows[name].append(abs(y_true - p) / (abs(y_true) + 1e-10) * 100)
                base_preds.append(p)

            if len(base_preds) == 3:
                p_stack = max(0.0, float(
                    meta_ridge.predict([[base_preds[0], base_preds[1], base_preds[2]]])[0]
                ))
                rows["stacking"].append(
                    abs(y_true - p_stack) / (abs(y_true) + 1e-10) * 100
                )

    mapes = {n: float(np.mean(v)) for n, v in rows.items() if v}
    print(f"  LightGBM: {mapes.get('lgbm', 0):.2f}%  "
          f"XGBoost: {mapes.get('xgb', 0):.2f}%  "
          f"ElasticNet: {mapes.get('en', 0):.2f}%")
    best_base = min(mapes.get("lgbm", np.inf),
                    mapes.get("xgb",  np.inf),
                    mapes.get("en",   np.inf))
    stack_mape = mapes.get("stacking", np.inf)
    delta      = stack_mape - best_base
    verdict    = "улучшает" if delta < -0.1 else ("нейтрально" if abs(delta) <= 0.1 else "не улучшает")
    print(f"  Стекинг:  {stack_mape:.2f}%  Δ vs лучшая={delta:+.2f}%  → {verdict}")

    _log(
        run_name="E8_stacking",
        params={"ridge_alpha": 1.0, "kfold_oof": kfold,
                "eval": "test no_exog h=1,3,6"},
        metrics={
            "lgbm_mape":       round(mapes.get("lgbm", 0), 3),
            "xgb_mape":        round(mapes.get("xgb", 0), 3),
            "en_mape":         round(mapes.get("en", 0), 3),
            "stacking_mape":   round(stack_mape, 3),
            "delta_vs_best":   round(delta, 3),
            "ridge_coef_lgbm": round(float(meta_ridge.coef_[0]), 3),
            "ridge_coef_xgb":  round(float(meta_ridge.coef_[1]), 3),
            "ridge_coef_en":   round(float(meta_ridge.coef_[2]), 3),
        },
        tags={"verdict": verdict, "note": "test_based"},
    )
    return mapes


# =============================================================================
# ИТОГОВОЕ СРАВНЕНИЕ
# =============================================================================

def run_all_experiments():
    """
    Запускает эксперименты E2-E8 для XGBoost VIF v2 модели.
    """
    print("=" * 65)
    print("ML v2 EXPERIMENTS — XGBoost VIF (5 переменных)")
    print("=" * 65)
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    print("\nЗагрузка артефактов финальной модели...")
    arts = load_production_artifacts()
    base_mape = arts["base_mape"]
    print(f"Базовый MAPE: {base_mape:.2f}%\n" if base_mape else "")

    results = {}

    for exp_fn, name in [
        (experiment_clustering,        "E2_clustering"),
        (experiment_cross_series,      "E3_cross_series"),
        (experiment_ablation,          "E4_ablation"),
        (experiment_horizon_split,     "E5_horizon_split"),
        (experiment_direct_vs_recursive, "E6_direct_recursive"),
        (experiment_global_vs_local,   "E7_global_local"),
        (experiment_stacking,          "E8_stacking"),
    ]:
        print()
        try:
            results[name] = exp_fn(arts)
        except Exception as e:
            print(f"  [ERROR] {name}: {e}")
            import traceback
            traceback.print_exc()

    print("\n\n" + "=" * 65)
    print("ИТОГОВАЯ ТАБЛИЦА ЭКСПЕРИМЕНТОВ")
    print(f"Базовый MAPE (XGBoost vif_exog no_exog): "
          f"{base_mape:.2f}%" if base_mape else "")
    print("=" * 65)

    summary_rows = []
    for name, res in results.items():
        if res is None or isinstance(res, pd.DataFrame):
            continue
        row = {"Эксперимент": name}
        for k, v in res.items():
            if isinstance(v, float):
                row[k] = round(v, 2)
        summary_rows.append(row)

    if summary_rows:
        print(pd.DataFrame(summary_rows).to_string(index=False))

    summary_path = ML_RESULTS_DIR / "ml_v2_experiments_summary.csv"
    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
        print(f"\n  → {summary_path}")

    print(f"\nMLflow: mlflow ui --backend-store-uri {MLFLOW_TRACKING_URI}")
    print("=" * 65)
    return results


if __name__ == "__main__":
    run_all_experiments()