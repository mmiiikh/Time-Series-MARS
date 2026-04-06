"""
ML модели для прогнозирования продаж Mars.

Классы:
  DirectForecaster — direct multi-step прогноз (отдельная модель на каждый горизонт).

Фабрики моделей:
  make_lgbm, make_xgb — создают модели с дефолтными или tuned параметрами.

Отбор признаков:
  select_features_lgbm   — LightGBM gain importance
  select_features_boruta — Boruta
  run_feature_selection  — полный pipeline отбора
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb

from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from boruta import BorutaPy

from src.config.settings import (
    TARGET_COL, RANDOM_STATE,
    LGBM_DEFAULT_PARAMS, XGB_DEFAULT_PARAMS, EN_DEFAULT_PARAMS,
)


# =============================================================================
# DIRECT FORECASTER
# =============================================================================

class DirectForecaster:
    """
    Direct multi-step прогноз.
    Обучается на log1p(y), predict возвращает expm1 (оригинальный масштаб).
    predict_log возвращает лог-масштаб (для quantile ДИ).

    Для каждого горизонта h обучается отдельная модель:
      X = признаки момента t
      y = log1p(продажи в момент t+h)
    """

    def __init__(self, model_fn, name: str, horizons: list):
        self.model_fn = model_fn
        self.name     = name
        self.horizons = horizons
        self.models   = {}

    def fit(self, X_train: pd.DataFrame, df_train: pd.DataFrame):
        for h in self.horizons:
            y_h   = (df_train
                     .groupby(["_category", "_channel"])[TARGET_COL]
                     .shift(-h))
            valid = y_h.notna()
            idx   = valid[valid].index
            self.models[h] = self.model_fn().fit(
                X_train.loc[idx], y_h.loc[idx]
            )
        return self

    def predict(self, X: pd.DataFrame, horizon: int) -> np.ndarray:
        """Оригинальный масштаб (expm1)."""
        return np.expm1(self.models[horizon].predict(X))

    def predict_log(self, X: pd.DataFrame, horizon: int) -> np.ndarray:
        """Лог-масштаб — для quantile ДИ."""
        return self.models[horizon].predict(X)


# =============================================================================
# ФАБРИКА МОДЕЛЕЙ — pickle-совместимая
# =============================================================================

class ModelFactory:
    """
    Callable класс для создания моделей с заданными параметрами.
    Определён на уровне модуля — pickle сериализует без проблем.

    Локальные функции (def внутри другой функции) pickle не сериализует.
    Этот класс решает проблему: DirectForecaster хранит его как model_fn
    и корректно сохраняется в pkl.

    Использование:
        factory = ModelFactory("lgbm", {"n_estimators": 500, ...})
        model   = factory()   # → LGBMRegressor(...)
    """

    def __init__(self, model_type: str, params: dict):
        self.model_type = model_type
        self.params     = params

    def __call__(self):
        if self.model_type == "lgbm":
            return lgb.LGBMRegressor(**self.params)
        elif self.model_type == "xgb":
            return xgb.XGBRegressor(**self.params)
        elif self.model_type == "elasticnet":
            return ElasticNetCV(**self.params)
        else:
            raise ValueError(f"Неизвестный тип модели: {self.model_type}")

    def __repr__(self):
        return f"ModelFactory({self.model_type}, {self.params})"



def make_lgbm(params: dict = None):
    """Фабрика LightGBM. Если params=None — дефолтные параметры."""
    p = params or LGBM_DEFAULT_PARAMS
    return lgb.LGBMRegressor(**p)


def make_xgb(params: dict = None):
    """Фабрика XGBoost. Если params=None — дефолтные параметры."""
    p = params or XGB_DEFAULT_PARAMS
    return xgb.XGBRegressor(**p)


def make_elasticnet(params: dict = None):
    """Фабрика ElasticNet. Если params=None — дефолтные параметры."""
    p = params or EN_DEFAULT_PARAMS
    return ElasticNetCV(**p)


def make_lgbm_quantile(alpha: float):
    """Quantile LightGBM для ДИ."""
    p = {**LGBM_DEFAULT_PARAMS, "objective": "quantile", "alpha": alpha}
    p.pop("verbose", None)
    return lgb.LGBMRegressor(**p, verbose=-1)


# =============================================================================
# ОТБОР ПРИЗНАКОВ
# =============================================================================

def select_features_lgbm(X_train: pd.DataFrame,
                          y_train: pd.Series) -> list:
    """LightGBM gain importance — убирает нулевые признаки."""
    model = lgb.LGBMRegressor(
        n_estimators=200, learning_rate=0.05, max_depth=5,
        num_leaves=31, min_child_samples=5,
        random_state=RANDOM_STATE, verbose=-1,
    )
    model.fit(X_train, y_train)
    imp      = pd.Series(model.feature_importances_, index=X_train.columns)
    selected = imp[imp > 0].index.tolist()
    print(f"  LightGBM gain: {len(X_train.columns)} → {len(selected)} признаков")
    return selected


def select_features_boruta(X_train: pd.DataFrame, y_train: pd.Series,
                            max_iter: int = 50) -> tuple:
    """Boruta — подтверждённые и пограничные признаки."""
    rf = RandomForestRegressor(
        n_estimators=100, max_depth=7,
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    boruta = BorutaPy(rf, n_estimators="auto", max_iter=max_iter,
                      random_state=RANDOM_STATE)
    boruta.fit(X_train.values, y_train.values)
    confirmed = X_train.columns[boruta.support_].tolist()
    tentative = X_train.columns[boruta.support_weak_].tolist()
    rejected  = X_train.columns[~boruta.support_].tolist()
    print(f"  Boruta: подтверждено {len(confirmed)}, "
          f"пограничных {len(tentative)}, отклонено {len(rejected)}")
    return confirmed, tentative


def run_feature_selection(global_df: pd.DataFrame,
                           feature_cols: list,
                           cv_folds: list) -> list:
    """
    Полный pipeline отбора признаков: LightGBM gain → Boruta.
    Запускается на train-части первого фолда.
    """
    train_mask, _, _ = cv_folds[0]
    X_train = global_df.loc[train_mask, feature_cols].fillna(0)
    y_train = global_df.loc[train_mask, TARGET_COL]

    print("\n=== ОТБОР ПРИЗНАКОВ ===")
    selected_lgbm        = select_features_lgbm(X_train, y_train)
    confirmed, tentative = select_features_boruta(
        X_train[selected_lgbm], y_train
    )
    final = confirmed + tentative

    # Принудительно включаем y_lag12 (критически важный лаг для сезонности)
    if "y_lag12" not in final and "y_lag12" in feature_cols:
        final.append("y_lag12")

    print(f"\n  Итого признаков: {len(final)}")
    return final


# =============================================================================
# CV MAPE SCORE (для Optuna и сравнения экспериментов)
# =============================================================================

def cv_mape_score(global_df: pd.DataFrame,
                  features: list,
                  cv_folds: list,
                  model_fn,
                  horizons: list = None) -> float:
    """
    Средний MAPE по CV фолдам и горизонтам.
    Используется в Optuna и для сравнения экспериментов.

    horizons по умолчанию [1, 3, 6] — h=12 нельзя добавить при test_size=12.
    """
    from src.config.settings import CV_HORIZONS
    from src.utils.metrics import compute_metrics

    if horizons is None:
        horizons = CV_HORIZONS

    mapes = []
    for train_mask, test_mask, _ in cv_folds:
        train_df = global_df.loc[train_mask].copy()
        test_df  = global_df.loc[test_mask].copy()
        X_train  = train_df[features].fillna(0)
        X_test   = test_df[features].fillna(0)

        for h in horizons:
            y_h_tr   = train_df.groupby(
                ["_category", "_channel"])[TARGET_COL].shift(-h)
            valid_tr = y_h_tr.notna()
            idx_tr   = valid_tr[valid_tr].index
            if len(idx_tr) == 0:
                continue

            y_h   = test_df.groupby(
                ["_category", "_channel"])[TARGET_COL].shift(-h)
            valid = y_h.notna()
            idx   = valid[valid].index
            if len(idx) == 0:
                continue

            y_true = np.expm1(y_h.loc[idx].values)
            model  = model_fn()
            model.fit(X_train.loc[idx_tr], y_h_tr.loc[idx_tr])
            y_pred = np.expm1(model.predict(X_test.loc[idx]))
            m      = compute_metrics(y_true, y_pred)
            if not np.isnan(m["mape"]):
                mapes.append(m["mape"])

    return float(np.mean(mapes)) if mapes else np.inf


# =============================================================================
# ЗАГРУЗКА СОХРАНЁННЫХ МОДЕЛЕЙ
# =============================================================================

def load_forecaster(pkl_path: str) -> "DirectForecaster":
    """
    Загружает сохранённый DirectForecaster из pkl файла.

    Формат pkl: {"models": {h: fitted_model}, "name": str, "horizons": list}
    Возвращает DirectForecaster с заполненным .models — готов к predict().

    Использование:
        fc = load_forecaster("models/ml/ml_lgbm_full.pkl")
        y_pred = fc.predict(X, horizon=3)
    """
    import pickle
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    # Совместимость: если сохранён старый формат (сам DirectForecaster)
    if isinstance(data, DirectForecaster):
        return data

    # Новый формат: словарь с models
    fc = DirectForecaster(
        model_fn=lambda: None,           # фабрика не нужна для predict
        name=data.get("name", "loaded"),
        horizons=data.get("horizons", list(data["models"].keys())),
    )
    fc.models = data["models"]
    return fc