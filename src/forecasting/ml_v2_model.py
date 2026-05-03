from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

try:
    import lightgbm as lgb
    import xgboost as xgb_module
    from sklearn.linear_model import ElasticNetCV
    from sklearn.ensemble import RandomForestRegressor
except ImportError:
    lgb = None
    xgb_module = None
    ElasticNetCV = None
    RandomForestRegressor = None

from src.config.settings import TARGET_COL, RANDOM_STATE

ARCH_NO_EXOG = "no_exog_features"
ARCH_NO_FLAGS = "exog_no_flags"
ARCH_WITH_FLAGS = "exog_with_flags"


def _safe(col: str) -> str:
    return col.replace(" ", "_").replace("/", "_").replace("-", "_")

def _has_flag(col: str) -> str:
    return f"has_{_safe(col)}"

def _future_col(col: str, h: int) -> str:
    return f"{_safe(col)}_future_h{h}"


class ModelFactoryV2:
    def __init__(self, model_type: str, params: dict):
        self.model_type = model_type
        self.params = params

    def __call__(self):
        if self.model_type == "lgbm":
            return lgb.LGBMRegressor(**self.params)
        elif self.model_type == "xgb":
            return xgb_module.XGBRegressor(**self.params)
        elif self.model_type == "elasticnet":
            return ElasticNetCV(**self.params)
        elif self.model_type == "rf":
            return RandomForestRegressor(**self.params)
        else:
            raise ValueError(
                f"Неизвестный тип модели: '{self.model_type}'. "
                f"Допустимые: lgbm, xgb, elasticnet, rf."
            )

    def __repr__(self):
        return f"ModelFactoryV2({self.model_type})"


def _apply_masking(X, exog_h_cols, selected_exog, last_value_row, mask_prob, rng):
    if not exog_h_cols or mask_prob <= 0:
        return X
    X_out = X.copy()
    n = len(X_out)
    for col in selected_exog:
        safe = _safe(col)
        future_for_col = [c for c in exog_h_cols if safe in c]
        hf = _has_flag(col)
        fill = last_value_row.get(col, 0.0)
        if not future_for_col:
            continue
        mask = rng.random(n) < mask_prob
        for fc in future_for_col:
            if fc in X_out.columns:
                X_out.loc[mask, fc] = fill
        if hf in X_out.columns:
            X_out.loc[mask, hf] = 0.0
    return X_out


class DirectForecasterV2:

    def __init__(
        self, model_fn, name, horizons, base_features,
        exog_future_cols_per_h, selected_exog,
        last_value_per_series, global_last,
        architecture=ARCH_WITH_FLAGS, mask_prob=0.3, scale_features=False):
        self.model_fn = model_fn
        self.name = name
        self.horizons = horizons
        self.base_features = base_features
        self.exog_future_cols_per_h = exog_future_cols_per_h
        self.selected_exog = selected_exog
        self.last_value_per_series  = last_value_per_series
        self.global_last = global_last
        self.architecture = architecture
        self.mask_prob = mask_prob
        self.scale_features = scale_features
        self.models:dict = {}
        self.scalers: dict = {}
        self.feature_names_per_h: dict = {}

    def _full_features(self, h):
        return self.base_features + self.exog_future_cols_per_h.get(h, [])

    def _get_last_value(self, cat, ch, col):
        return self.last_value_per_series.get((cat, ch), {}).get(col, self.global_last.get(col, 0.0))

    def fit(self, df_train: pd.DataFrame) -> "DirectForecasterV2":
        rng = np.random.default_rng(RANDOM_STATE)

        for h in self.horizons:
            features_h  = self._full_features(h)
            exog_h_cols = self.exog_future_cols_per_h.get(h, [])
            y_h = df_train.groupby(["_category", "_channel"])[TARGET_COL].shift(-h)
            valid = y_h.notna()
            for fc in exog_h_cols:
                if fc in df_train.columns:
                    valid = valid & df_train[fc].notna()
            idx = valid[valid].index
            if len(idx) == 0:
                continue

            avail = [f for f in features_h if f in df_train.columns]
            X_h = df_train.loc[idx, avail].fillna(0).copy()
            y_h_fit = y_h.loc[idx]
            if self.architecture == ARCH_WITH_FLAGS and exog_h_cols:
                X_h_masked = X_h.copy()
                for (cat, ch), grp_idx in df_train.groupby(["_category", "_channel"]).groups.items():
                    rows_in_X = X_h.index.intersection(grp_idx)
                    if len(rows_in_X) == 0:
                        continue
                    series_last = {col: self._get_last_value(cat, ch, col) for col in self.selected_exog}
                    X_h_masked.loc[rows_in_X] = _apply_masking(
                        X_h.loc[rows_in_X].copy(),
                        exog_h_cols, self.selected_exog,
                        series_last, self.mask_prob, rng)
                X_h = X_h_masked

            if self.scale_features:
                scaler  = StandardScaler()
                X_h_fit = pd.DataFrame(scaler.fit_transform(X_h), columns=avail, index=X_h.index)
                self.scalers[h] = scaler
            else:
                X_h_fit = X_h

            self.models[h] = self.model_fn().fit(X_h_fit, y_h_fit)
            self.feature_names_per_h[h] = avail
        return self

    def _build_x_pred(self, x_base, h, cat, ch, user_exog):
        feats_h = self.feature_names_per_h.get(h, self._full_features(h))
        exog_h_cols = self.exog_future_cols_per_h.get(h, [])
        X_pred = pd.DataFrame(0.0, index=[0], columns=feats_h)

        for col in feats_h:
            if col in x_base.columns and col not in exog_h_cols:
                X_pred[col] = x_base[col].values[0]

        if self.architecture == ARCH_NO_EXOG:
            pass

        elif self.architecture == ARCH_NO_FLAGS:
            for orig_col in self.selected_exog:
                fc = _future_col(orig_col, h)
                if fc not in feats_h:
                    continue
                X_pred[fc] = (float(user_exog[orig_col]) if user_exog and orig_col in user_exog
                    else self._get_last_value(cat, ch, orig_col))

        else:
            for orig_col in self.selected_exog:
                fc = _future_col(orig_col, h)
                hf = _has_flag(orig_col)
                if fc not in feats_h:
                    continue
                if user_exog and orig_col in user_exog:
                    X_pred[fc] = float(user_exog[orig_col])
                    if hf in feats_h:
                        X_pred[hf] = 1.0
                else:
                    X_pred[fc] = self._get_last_value(cat, ch, orig_col)
                    if hf in feats_h:
                        X_pred[hf] = 0.0

        if self.scale_features and h in self.scalers:
            X_pred = pd.DataFrame(self.scalers[h].transform(X_pred), columns=feats_h)
        return X_pred

    def predict(self, x_base, h, cat=None, ch=None, user_exog=None):
        if h not in self.models:
            h = max(self.models.keys())
        return np.expm1(self.models[h].predict(self._build_x_pred(x_base, h, cat, ch, user_exog)))