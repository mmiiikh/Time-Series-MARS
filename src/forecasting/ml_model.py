"""
первая версия ML моделей с отобор признаков, построением 4 моделей и оформлением в фабрики
"""

import pickle
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from boruta import BorutaPy
from src.config.settings import TARGET_COL, RANDOM_STATE,LGBM_DEFAULT_PARAMS, XGB_DEFAULT_PARAMS,EN_DEFAULT_PARAMS, RF_DEFAULT_PARAMS


class DirectForecaster:
    """
    direct multi-step прогноз
    обучается на log1p(y), возвращает expm1
    """
    def __init__(self, model_fn, name: str, horizons: list):
        self.model_fn = model_fn
        self.name = name
        self.horizons = horizons
        self.models = {}

    def fit(self, X_train: pd.DataFrame, df_train: pd.DataFrame):
        for h in self.horizons:
            y_h = (df_train.groupby(["_category", "_channel"])[TARGET_COL].shift(-h))
            valid = y_h.notna()
            idx = valid[valid].index
            self.models[h] = self.model_fn().fit(X_train.loc[idx], y_h.loc[idx])
        return self

    def predict(self, X: pd.DataFrame, horizon: int) -> np.ndarray:
        return np.expm1(self.models[horizon].predict(X))

    def predict_log(self, X: pd.DataFrame, horizon: int) -> np.ndarray:
        return self.models[horizon].predict(X)


class ModelFactory:
    """
    класс для создания моделей с заданными параметрами, решает проблему сериализации в pkl:DirectForecaster хранит
    его как model_fn и корректно сохраняет в pkl.
    """

    def __init__(self, model_type: str, params: dict):
        self.model_type = model_type
        self.params = params

    def __call__(self):
        if self.model_type == "lgbm":
            return lgb.LGBMRegressor(**self.params)
        elif self.model_type == "xgb":
            return xgb.XGBRegressor(**self.params)
        elif self.model_type == "rf":
            return RandomForestRegressor(**self.params)
        elif self.model_type == "elasticnet":
            return ElasticNetCV(**self.params)

    def __repr__(self):
        return f"ModelFactory({self.model_type}, {self.params})"


def make_lgbm(params: dict = None) -> lgb.LGBMRegressor:
    p = params or LGBM_DEFAULT_PARAMS
    return lgb.LGBMRegressor(**p)

def make_xgb(params: dict = None) -> xgb.XGBRegressor:
    p = params or XGB_DEFAULT_PARAMS
    return xgb.XGBRegressor(**p)


def make_rf(params: dict = None) -> RandomForestRegressor:
    p = params or RF_DEFAULT_PARAMS
    return RandomForestRegressor(**p)


def make_elasticnet(params: dict = None) -> ElasticNetCV:
    p = params or EN_DEFAULT_PARAMS
    return ElasticNetCV(**p)


def make_lgbm_quantile(alpha: float) -> lgb.LGBMRegressor:
    """
    quantile LGBM для построения ди
    """
    p = {**LGBM_DEFAULT_PARAMS, "objective": "quantile", "alpha": alpha}
    p.pop("verbose", None)
    return lgb.LGBMRegressor(**p, verbose=-1)


def select_features_lgbm(X_train: pd.DataFrame,y_train: pd.Series) -> list:
    """
    LGBM gain importance,убирает признаки с нулевой важностью
    """
    model = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05, max_depth=5,num_leaves=31, min_child_samples=5,random_state=RANDOM_STATE, verbose=-1)
    model.fit(X_train, y_train)
    imp = pd.Series(model.feature_importances_, index=X_train.columns)
    selected = imp[imp > 0].index.tolist()
    print(f"LightGBM gain: {len(X_train.columns)}, {len(selected)} признаков")
    return selected


def select_features_boruta(X_train: pd.DataFrame,y_train: pd.Series,max_iter: int = 50) -> tuple:
    """
    Boruta выбор подтверждённых и пограничных признаков
    RF с Shadow features
    """
    rf = RandomForestRegressor(n_estimators=100, max_depth=7,random_state=RANDOM_STATE, n_jobs=-1)
    boruta = BorutaPy(rf, n_estimators="auto", max_iter=max_iter,random_state=RANDOM_STATE)
    boruta.fit(X_train.values, y_train.values)
    confirmed = X_train.columns[boruta.support_].tolist()
    tentative = X_train.columns[boruta.support_weak_].tolist()
    rejected = X_train.columns[~boruta.support_].tolist()
    print(f"подтверждено {len(confirmed)},пограничных {len(tentative)},отклонено {len(rejected)}")
    return confirmed, tentative


def run_feature_selection(global_df: pd.DataFrame,feature_cols: list,cv_folds: list) -> list:
    """
    полный pipeline отбора признаков
    """
    train_mask, _, _ = cv_folds[0]
    X_train = global_df.loc[train_mask, feature_cols].fillna(0)
    y_train = global_df.loc[train_mask, TARGET_COL]
    selected_lgbm = select_features_lgbm(X_train, y_train)
    confirmed, tentative = select_features_boruta(X_train[selected_lgbm], y_train)
    final = confirmed+tentative
    if "y_lag12" not in final and "y_lag12" in feature_cols:
        final.append("y_lag12")
    print(f"итог:{len(final)}")
    return final


def cv_mape_score(global_df: pd.DataFrame,
                  features: list,
                  cv_folds: list,
                  model_fn,
                  horizons: list = None) -> float:

    from src.config.settings import CV_HORIZONS
    from src.utils.metrics import compute_metrics
    if horizons is None:
        horizons = CV_HORIZONS
    mapes = []
    for train_mask, test_mask, _ in cv_folds:
        train_df = global_df.loc[train_mask].copy()
        test_df = global_df.loc[test_mask].copy()
        X_train = train_df[features].fillna(0)
        X_test = test_df[features].fillna(0)
        for h in horizons:
            y_h_tr = train_df.groupby(["_category", "_channel"])[TARGET_COL].shift(-h)
            valid_tr = y_h_tr.notna()
            idx_tr = valid_tr[valid_tr].index
            if len(idx_tr) == 0:
                continue

            y_h = test_df.groupby(["_category", "_channel"])[TARGET_COL].shift(-h)
            valid = y_h.notna()
            idx = valid[valid].index
            if len(idx) == 0:
                continue
            y_true = np.expm1(y_h.loc[idx].values)
            model = model_fn()
            model.fit(X_train.loc[idx_tr], y_h_tr.loc[idx_tr])
            y_pred = np.expm1(model.predict(X_test.loc[idx]))
            m = compute_metrics(y_true, y_pred)
            if not np.isnan(m["mape"]):
                mapes.append(m["mape"])
    return float(np.mean(mapes)) if mapes else np.inf


def load_forecaster(pkl_path: str) -> DirectForecaster:
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    if isinstance(data, DirectForecaster):
        return data
    fc = DirectForecaster(model_fn=lambda: None,name=data.get("name", "loaded"),horizons=data.get("horizons", list(data["models"].keys())))
    fc.models = data["models"]
    return fc