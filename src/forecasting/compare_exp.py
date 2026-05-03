"""
одноразовый скрипт, проверка двух экспериментов
"""
import json
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from src.config.settings import TARGET_COL, RANDOM_STATE, TEST_SIZE, HORIZON,ML_MODELS_DIR, ML_DATA_FILE,XGB_DEFAULT_PARAMS
from src.data.load_data import load_data, create_series_dict
from src.data.preprocess import build_global_dataset
from src.forecasting.ml_v2_model import DirectForecasterV2, ModelFactoryV2,ARCH_NO_FLAGS, _future_col, _safe

PROD_CONFIG = "vif_exog"
SHORT_HS    = [1,2,3]
LONG_HS     = list(range(4, HORIZON+1))

def load_artifacts() -> dict:
    meta_path = ML_MODELS_DIR/f"ml_v2_{PROD_CONFIG}_metadata.json"
    with open(meta_path) as f:
        meta = json.load(f)
    tuned_path = ML_MODELS_DIR/f"ml_v2_{PROD_CONFIG}_tuned_params.json"
    if tuned_path.exists():
        with open(tuned_path) as f:
            tuned = json.load(f)
    else:
        tuned = {"xgb":XGB_DEFAULT_PARAMS}

    df_raw = load_data(str(ML_DATA_FILE))
    series_dict = create_series_dict(df_raw)
    lag_ranges = meta.get("ccf_lag_ranges", {})
    global_df, _, _, _ = build_global_dataset(series_dict, lag_ranges)
    selected_exog = meta["selected_exog"]
    horizons = list(range(1, HORIZON+1))
    df = global_df.copy()
    exog_frames = []
    for (cat, ch), raw_df in series_dict.items():
        cols_here = [c for c in selected_exog if c in raw_df.columns]
        if not cols_here:
            continue
        tmp = raw_df[cols_here].copy()
        tmp["_category"] = cat
        tmp["_channel"] = ch
        tmp["_date"] = tmp.index
        exog_frames.append(tmp.reset_index(drop=True))

    if exog_frames:
        exog_merged = pd.concat(exog_frames, ignore_index=True)
        exog_merged["_date"] = pd.to_datetime(exog_merged["_date"])
        df["_date"] = pd.to_datetime(df["_date"])
        df = df.merge(exog_merged, on=["_category", "_channel", "_date"], how="left", suffixes=("", "_raw")).reset_index(drop=True)

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

    lv_json = meta.get("last_value_per_series", {})
    last_value_per_series = {tuple(k.split("|", 1)): v for k, v in lv_json.items()}
    global_last = meta.get("global_last", {})
    xgb_p = {**tuned.get("xgb", XGB_DEFAULT_PARAMS),"verbosity": 0, "random_state": RANDOM_STATE}
    dates = sorted(df["_date"].unique())
    train_df = df[df["_date"].isin(dates[:-TEST_SIZE])].copy()
    test_df= df[df["_date"].isin(dates[-TEST_SIZE:])].copy()

    print(f"Config:{PROD_CONFIG}|ExoG: {selected_exog}")
    print(f"XGBoost params: n_estimators={xgb_p.get('n_estimators')}"
          f"max_depth={xgb_p.get('max_depth')}"
          f"reg_lambda={xgb_p.get('reg_lambda', 1.0)}")
    print(f"Train: {len(train_df)} строк Test: {len(test_df)} строк")

    return {
        "global_df_v2":           df,
        "train_df":               train_df,
        "test_df":                test_df,
        "series_dict":            series_dict,
        "base_features":          meta["base_features"],
        "selected_exog":          selected_exog,
        "exog_future_cols_per_h": exog_future_cols_per_h,
        "last_value_per_series":  last_value_per_series,
        "global_last":            global_last,
        "architecture":           meta["architecture"],
        "mask_prob":              meta.get("mask_prob", 0.3),
        "xgb_p":                  xgb_p}


def evaluate_h12(model_name:str,fc_model: DirectForecasterV2,train_df: pd.DataFrame,test_df: pd.DataFrame,base_features:list) -> pd.DataFrame:
    rows = []
    for (cat, ch), grp_te in test_df.groupby(["_category", "_channel"]):
        tr_s = train_df[(train_df["_category"] == cat) & (train_df["_channel"] == ch)]
        x_T = tr_s[base_features].fillna(0).iloc[[-1]].copy()
        h_preds, h_trues = [], []
        for h, fd_str in enumerate(sorted(grp_te["_date"].unique()),start=1):
            if h not in fc_model.models:
                continue
            fact = grp_te[grp_te["_date"] == fd_str]
            if fact.empty:
                continue
            y_true = float(np.expm1(fact[TARGET_COL].values[0]))
            x_h = x_T.copy()
            fd = pd.Timestamp(fd_str)
            for feat, val in [
                ("month",int(fd.month)),
                ("month_sin", float(np.sin(2*np.pi*fd.month/12))),
                ("month_cos", float(np.cos(2*np.pi*fd.month/12))),
                ("quarter", int(fd.quarter)),
                ("quarter_sin", float(np.sin(2*np.pi*fd.quarter/4))),
                ("quarter_cos", float(np.cos(2*np.pi*fd.quarter/4))),
                ("is_q4", int(fd.month>=10)),
                ("is_summer", int(fd.month in [6,7,8])),
                ("covid",0), ("post_covid",0)]:
                if feat in base_features:
                    x_h[feat] = val
            if "t" in base_features and "t" in x_T.columns:
                new_t = float(x_T["t"].values[0])+h
                x_h["t"] = new_t
                if "t_squared" in base_features:
                    x_h["t_squared"] = new_t**2

            y_pred = max(0.0, float(fc_model.predict(x_h, h, cat=cat, ch=ch, user_exog=None)[0]))
            h_preds.append(y_pred)
            h_trues.append(y_true)

        if not h_trues:
            continue

        y_t = np.array(h_trues)
        y_p = np.array(h_preds)
        mape = float(np.mean(np.abs(y_t - y_p) / (np.abs(y_t) + 1e-10)) * 100)
        mae = float(np.mean(np.abs(y_t - y_p)))
        rows.append({"model":model_name, "category":cat,"channel":ch,"mape": round(mape, 2),"mae": round(mae, 2)})

    return pd.DataFrame(rows)


def train_unified(arts: dict) -> DirectForecasterV2:
    kw = _kw(arts, list(range(1, HORIZON+1)))
    fc = DirectForecasterV2(model_fn=ModelFactoryV2("xgb", arts["xgb_p"]),name="unified", **kw)
    print("  Обучение Unified XGBoost...", end=" ", flush=True)
    fc.fit(arts["train_df"])
    print("OK")
    return fc

def train_horizon_split(arts: dict) -> tuple[DirectForecasterV2, DirectForecasterV2]:

    xgb_p = arts["xgb_p"]
    base_lambda = xgb_p.get("reg_lambda", 1.0)
    p_short = {**xgb_p, "reg_lambda": max(0.1, base_lambda * 0.3)}
    p_long  = {**xgb_p, "reg_lambda": min(10.0, base_lambda * 3.0)}
    kw_short = _kw(arts, SHORT_HS)
    kw_long  = _kw(arts, LONG_HS)
    fc_short = DirectForecasterV2(model_fn=ModelFactoryV2("xgb", p_short),name="split_short", **kw_short)
    fc_long = DirectForecasterV2(model_fn=ModelFactoryV2("xgb", p_long),name="split_long", **kw_long)
    fc_short.fit(arts["train_df"])
    print("OK")
    fc_long.fit(arts["train_df"])
    print("OK")
    return fc_short, fc_long


def _kw(arts: dict, horizons: list) -> dict:
    return dict(
        horizons=horizons,
        base_features=arts["base_features"],
        exog_future_cols_per_h={h: arts["exog_future_cols_per_h"].get(h, []) for h in horizons},
        selected_exog=arts["selected_exog"],
        last_value_per_series=arts["last_value_per_series"],
        global_last=arts["global_last"],
        architecture=arts["architecture"],
        mask_prob=arts["mask_prob"])


class HorizonSplitWrapper:
    def __init__(self, fc_short, fc_long):
        self.fc_short = fc_short
        self.fc_long = fc_long
        self.models = {**fc_short.models, **fc_long.models}

    def predict(self, x_h, h, cat=None, ch=None, user_exog=None):
        fc = self.fc_short if h in SHORT_HS else self.fc_long
        return fc.predict(x_h, h, cat=cat, ch=ch, user_exog=user_exog)


def main():
    arts = load_artifacts()
    fc_unified = train_unified(arts)
    fc_short, fc_long = train_horizon_split(arts)
    fc_split = HorizonSplitWrapper(fc_short, fc_long)
    df_unified = evaluate_h12("Unified", fc_unified, arts["train_df"],arts["test_df"], arts["base_features"])
    df_split = evaluate_h12("Horizon-Split", fc_split,  arts["train_df"],arts["test_df"], arts["base_features"])
    df_all = pd.concat([df_unified, df_split], ignore_index=True)
    pivot = df_all.pivot_table(index=["category", "channel"],columns="model",values="mape").reset_index()
    pivot.columns.name = None
    pivot["Δ (Split-Unified)"] = (pivot.get("Horizon-Split", np.nan) - pivot.get("Unified", np.nan)).round(2)
    pivot = pivot.sort_values("Δ (Split-Unified)")
    print(pivot.to_string(index=False))
    med_unified = df_unified["mape"].median()
    med_split = df_split["mape"].median()
    delta = med_split - med_unified
    win_split = (pivot["Δ (Split-Unified)"] < 0).sum()
    win_unified = (pivot["Δ (Split-Unified)"] > 0).sum()
    tie = (pivot["Δ (Split-Unified)"] == 0).sum()

    print(f" {'Unified XGBoost:':<35} {med_unified:.2f}%")
    print(f" {'Horizon-Split XGBoost:':<35} {med_split:.2f}% ")
    out = pivot.copy()
    out.to_csv("compare_horizon_split_results.csv", index=False)
    return pivot

if __name__ == "__main__":
    main()