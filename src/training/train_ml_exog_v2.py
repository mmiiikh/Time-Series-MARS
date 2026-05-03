"""
ML модель с опциональными экзогенными переменными.
"""

from __future__ import annotations
import json
import pickle
import warnings
import numpy as np
import pandas as pd
import mlflow
import lightgbm as lgb
import xgboost as xgb_module
import optuna
from sklearn.ensemble import RandomForestRegressor
from numpy.linalg import lstsq
from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import StandardScaler

from src.config.settings import (
    ML_DATA_FILE, ML_MODELS_DIR, ML_RESULTS_DIR,
    MLFLOW_TRACKING_URI,
    MLFLOW_EXPERIMENT,
    TARGET_COL, EXOG_COLS, RANDOM_STATE,
    TEST_SIZE, N_FOLDS, HORIZON, CV_HORIZONS,
    LGBM_DEFAULT_PARAMS, XGB_DEFAULT_PARAMS, EN_DEFAULT_PARAMS, RF_DEFAULT_PARAMS,
    OPTUNA_N_TRIALS_RF)
from src.data.load_data import load_data, create_series_dict
from src.data.preprocess import build_global_dataset, make_cv_folds
from src.utils.metrics import compute_metrics

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)
np.random.seed(RANDOM_STATE)
MLFLOW_EXPERIMENT_V2 = "mars_ml_exog_v2"
MLFLOW_EXPERIMENT_V2_COMPARE = "mars_ml_exog_v2_comparison"
CCF_THRESHOLD = 0.15
MASK_PROB_CANDIDATES = [0.2,0.3,0.5]
OPTUNA_N_TRIALS_V2 = 25

from src.forecasting.ml_v2_model import (
    DirectForecasterV2,
    ModelFactoryV2,
    ARCH_NO_EXOG,
    ARCH_NO_FLAGS,
    ARCH_WITH_FLAGS,
    _safe,
    _has_flag,
    _future_col,
    _apply_masking)

PKL_NAMES = {
    "LightGBM":    ("ml_v2_lgbm_eval.pkl", "ml_v2_lgbm_full.pkl"),
    "XGBoost":     ("ml_v2_xgb_eval.pkl", "ml_v2_xgb_full.pkl"),
    "RandomForest": ("ml_v2_rf_eval.pkl", "ml_v2_rf_full.pkl"),
    "ElasticNet":  ("ml_v2_en_eval.pkl", "ml_v2_en_full.pkl")}


PROD_CONFIG = "vif_exog"
PROD_MODEL = "XGBoost"


def step_load(lag_ranges=None):
    if lag_ranges is None:
        for fname in ["ml_metadata.json", "ml_v2_metadata.json"]:
            p = ML_MODELS_DIR / fname
            if p.exists():
                with open(p) as f:
                    lag_ranges = json.load(f).get("ccf_lag_ranges", {})
                print(f"  lag_ranges из {fname}")
                break

    if lag_ranges is None:
        from src.data.preprocess import analyze_ccf, build_ccf_lag_ranges
        df_tmp = load_data(str(ML_DATA_FILE))
        sd_tmp = create_series_dict(df_tmp)
        _, ccf_sum = analyze_ccf(sd_tmp)
        lag_ranges = build_ccf_lag_ranges(ccf_sum)
        print("lag_ranges пересчитаны через CCF")

    df_raw = load_data(str(ML_DATA_FILE))
    series_dict = create_series_dict(df_raw)
    global_df, feature_cols, cat_enc, ch_enc = build_global_dataset(series_dict, lag_ranges)
    print(f"  Рядов: {len(series_dict)}  |  Точек: {len(global_df)}")
    return df_raw, series_dict, global_df, feature_cols, cat_enc, ch_enc, lag_ranges


def step_ccf_exog_selection(series_dict: dict,global_df: pd.DataFrame,threshold: float = CCF_THRESHOLD) -> list[str]:
    dates = sorted(global_df["_date"].unique())
    train_cutoff = max(set(dates[:-TEST_SIZE]))
    print(f"  Train период: до {train_cutoff}")

    avail = [col for col in EXOG_COLS
        if any(
            col in df.columns and df[col].notna().sum() > 0
            for df in series_dict.values())]
    print(f"Доступных: {len(avail)} из {len(EXOG_COLS)}")
    if not avail:
        print("[ERROR] Ни одна EXOG переменная не найдена в series_dict!")
        sample_key = next(iter(series_dict))
        print(f"Колонки примера: {list(series_dict[sample_key].columns)}")
        return []
    results = {}
    for col in avail:
        max_ccfs = []
        for (cat, ch), df in series_dict.items():
            if col not in df.columns:
                continue
            df_train = df[df.index <= pd.Timestamp(train_cutoff)]
            y = df_train[TARGET_COL].dropna()
            x = df_train[col].dropna()
            common = y.index.intersection(x.index)
            if len(common) < 24:
                continue
            y_diff = np.diff(y[common].values)
            x_diff = np.diff(x[common].values)
            if len(y_diff) < 5:
                continue
            y_s = (y_diff-y_diff.mean())/(y_diff.std()+1e-10)
            x_s = (x_diff-x_diff.mean())/(x_diff.std()+1e-10)
            best = 0.0
            for lag in range(0, 13):
                n = len(y_s)-lag
                if n > 0:
                    c = float(np.corrcoef(y_s[lag:], x_s[:n])[0, 1])
                    if not np.isnan(c):
                        best = max(best, abs(c))
            max_ccfs.append(best)
        results[col] = float(np.median(max_ccfs)) if max_ccfs else 0.0

    sorted_cols = sorted(results.items(), key=lambda x: -x[1])
    selected = []
    for col, val in sorted_cols:
        flag = "ok" if val >= threshold else "not ok"
        print(f"  {col:<40} {val:>10.4f}  {flag:>10}")
        if val >= threshold:
            selected.append(col)

    if not selected:
        print(f"Нет переменных выше {threshold},берём топ-3")
        selected = [col for col, _ in sorted_cols[:3]]
    print(f"Отобрано: {len(selected)}: {selected}")
    return selected


def step_vif_exog_selection(series_dict: dict,global_df: pd.DataFrame,selected_exog: list[str],vif_threshold: float = 5.0) -> list[str]:
    if len(selected_exog) <= 1:
        print("Пропускаем")
        return list(selected_exog)

    dates = sorted(global_df["_date"].unique())
    train_cutoff = max(set(dates[:-TEST_SIZE]))
    series_means = {}
    for col in selected_exog:
        col_means = []
        for (cat, ch), raw_df in series_dict.items():
            if col not in raw_df.columns:
                continue
            s = raw_df[raw_df.index <= pd.Timestamp(train_cutoff)][col].dropna()
            if len(s) > 0:
                col_means.append(float(s.mean()))
        series_means[col] = col_means

    min_len = min(len(v) for v in series_means.values())
    if min_len < 3:
        print("Мало рядов")
        return list(selected_exog)

    df_vif = pd.DataFrame({col: vals[:min_len] for col, vals in series_means.items()})
    print(f"Рядов в VIF-матрице: {len(df_vif)}")
    remaining = list(selected_exog)  # сохраняем CCF-порядок
    for iteration in range(1, 20):
        if len(remaining) <= 1:
            break

        X = df_vif[remaining].values.astype(float)
        vifs = {}
        for i, col in enumerate(remaining):
            X_other = np.delete(X, i, axis=1)
            if X_other.shape[1] == 0:
                vifs[col] = 1.0
                continue
            try:
                A = np.column_stack([np.ones(len(X_other)), X_other])
                coef, _, _, _ = lstsq(A, X[:, i], rcond=None)
                y_pred = A @ coef
                ss_res = float(np.sum((X[:, i]-y_pred)**2))
                ss_tot = float(np.sum((X[:, i]-X[:, i].mean())**2))
                r2 = 1.0-ss_res/ss_tot if ss_tot>0 else 0.0
                r2 = min(max(r2, 0.0), 0.9999)
                vifs[col] = round(1.0/(1.0-r2), 2)
            except Exception:
                vifs[col] = 1.0

        max_col = max(vifs, key=vifs.get)
        max_vif = vifs[max_col]

        print(f"Итерация {iteration}:")
        for col in remaining:
            flag = "delete" if col == max_col and max_vif > vif_threshold else ""
            print(f"{col:<40} VIF={vifs[col]:.2f}{flag}")

        if max_vif <= vif_threshold:
            print(f"Все VIF ≤ {vif_threshold}")
            break
        remaining.remove(max_col)

    print(f"После VIF: {len(remaining)}: {remaining}")
    return remaining


def step_add_future_exog(global_df: pd.DataFrame,series_dict: dict,selected_exog: list[str],horizons: list[int]) -> tuple[pd.DataFrame, dict, list, list, dict, dict]:
    dates = sorted(global_df["_date"].unique())
    train_cutoff = max(set(dates[:-TEST_SIZE]))
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
        df = df.merge(exog_merged,on=["_category", "_channel", "_date"],how="left",suffixes=("", "_raw"))
        df = df.reset_index(drop=True)
        found = [c for c in selected_exog if c in df.columns]
        print(f"Смёржено: {len(found)}: {found}")

    last_value_per_series: dict[tuple, dict[str, float]] = {}
    for (cat, ch), raw_df in series_dict.items():
        raw_train = raw_df[raw_df.index <= pd.Timestamp(train_cutoff)]
        lv = {}
        for col in selected_exog:
            if col in raw_train.columns:
                s = raw_train[col].dropna()
                lv[col] = float(s.iloc[-1]) if len(s) > 0 else 0.0
            else:
                lv[col] = 0.0
        last_value_per_series[(cat, ch)] = lv

    global_last: dict[str, float] = {}
    for col in selected_exog:
        vals = [v.get(col, 0.0) for v in last_value_per_series.values()]
        global_last[col] = float(np.median(vals)) if vals else 0.0

    print(f"last_value вычислены для {len(last_value_per_series)} рядов")

    exog_future_cols_per_h: dict[int, list] = {}
    all_future_cols: list = []
    for h in horizons:
        h_cols = []
        for col in selected_exog:
            if col not in df.columns:
                continue
            fc = _future_col(col, h)
            df[fc] = df.groupby(["_category", "_channel"])[col].shift(-h)
            h_cols.append(fc)
            if fc not in all_future_cols:
                all_future_cols.append(fc)
        exog_future_cols_per_h[h] = h_cols

    has_flag_cols = []
    for col in selected_exog:
        if col in df.columns:
            hf = _has_flag(col)
            df[hf] = 1.0
            has_flag_cols.append(hf)

    print(f"future-exog колонок: {len(all_future_cols)}")
    print(f"has-flags: {has_flag_cols}")
    return df, exog_future_cols_per_h, all_future_cols, has_flag_cols, last_value_per_series, global_last


def step_leakage_check(global_df_v2, exog_future_cols_per_h, selected_exog):
    dates = sorted(global_df_v2["_date"].unique())
    test_months = set(str(d)[:7] for d in dates[-TEST_SIZE:])
    train_dates = set(dates[:-TEST_SIZE])
    train_df = global_df_v2[global_df_v2["_date"].isin(train_dates)].copy()
    leakage = False
    for h, h_cols in exog_future_cols_per_h.items():
        y_h = train_df.groupby(["_category", "_channel"])[TARGET_COL].shift(-h)
        for fc in h_cols:
            if fc not in train_df.columns:
                continue
            will_train = y_h.notna() & train_df[fc].notna()
            if will_train.sum() == 0:
                continue
            future_months = (pd.to_datetime(train_df[will_train]["_date"]) + pd.DateOffset(months=h)).dt.strftime("%Y-%m")
            if future_months.apply(lambda m: m in test_months).any():
                print(f"h={h}, col={fc}: утечка")
                leakage = True

    if leakage:
        raise AssertionError("Утечка данных")
    print(f"Утечки нет. Горизонтов={len(exog_future_cols_per_h)} переменных={len(selected_exog)}")


def _load_tuned_params() -> dict:
    path = ML_MODELS_DIR / "ml_metadata.json"
    if path.exists():
        with open(path) as f:
            meta = json.load(f)
        return {
            "lgbm": meta.get("lgbm_params", LGBM_DEFAULT_PARAMS),
            "xgb": meta.get("xgb_params", XGB_DEFAULT_PARAMS),
            "rf": meta.get("rf_params", RF_DEFAULT_PARAMS)}
    return {"lgbm": LGBM_DEFAULT_PARAMS, "xgb": XGB_DEFAULT_PARAMS, "rf": RF_DEFAULT_PARAMS}


def step_select_features_v2(global_df_v2, feature_cols, exog_future_cols_per_h,
    all_future_cols, has_flag_cols,
    selected_exog_ccf_order: list) -> tuple[list, dict, list]:
    baseline_meta = ML_MODELS_DIR / "ml_metadata.json"
    if baseline_meta.exists():
        with open(baseline_meta) as f:
            meta = json.load(f)
        base_raw = [f for f in meta["final_features"] if f in global_df_v2.columns]
        print(f"Базовых из ml_metadata.json: {len(base_raw)}")
    else:
        excluded = set(all_future_cols) | set(has_flag_cols)
        base_raw = [f for f in feature_cols if f in global_df_v2.columns and f not in excluded]
        print(f"Базовых (fallback): {len(base_raw)}")
    dates = sorted(global_df_v2["_date"].unique())
    train_df = global_df_v2[global_df_v2["_date"].isin(set(dates[:-TEST_SIZE]))].copy()
    nonzero_overall: set[str] = set()
    for h_test in [3, 6]:
        exog_h = [c for c in exog_future_cols_per_h.get(h_test, []) if c in train_df.columns]
        all_feats = [f for f in base_raw + exog_h if f in train_df.columns and not f.startswith("has_")]
        y_h = train_df.groupby(["_category", "_channel"])[TARGET_COL].shift(-h_test)
        valid = y_h.notna()
        for fc in exog_h:
            valid = valid & train_df[fc].notna()
        idx = valid[valid].index

        if len(idx) < 50:
            nonzero_overall.update(f for f in base_raw if not f.startswith("has_"))
            continue

        model_gain = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, max_depth=5,
            num_leaves=31, random_state=RANDOM_STATE, verbose=-1)
        model_gain.fit(train_df.loc[idx, all_feats].fillna(0), y_h.loc[idx])
        imp = pd.Series(model_gain.feature_importances_, index=all_feats)
        nonzero_h = set(imp[imp > 0].index.tolist())
        nonzero_overall.update(nonzero_h)
        zero_feats = [f for f in all_feats if f not in nonzero_h]
        print(f" h={h_test}: {len(all_feats)} признаков: {len(nonzero_h)} ненулевых")
        print(f"Нулевых: {zero_feats}")
        exog_imp = imp[[f for f in exog_h if f in imp.index]].sort_values(ascending=False)
        if len(exog_imp) > 0:
            print(f"Future-exog importance: {exog_imp.to_dict()}")

    final_base = [f for f in base_raw if f in nonzero_overall and not f.startswith("has_")]
    if "y_lag12" in base_raw and "y_lag12" not in final_base:
        final_base.append("y_lag12")

    good_exog_vars: set[str] = set()
    for fc in nonzero_overall:
        if "future_h" not in fc:
            continue
        for col in EXOG_COLS:
            if _safe(col) in fc:
                good_exog_vars.add(col)

    if not good_exog_vars:
        good_exog_vars = set(col for h_cols in exog_future_cols_per_h.values()
            for fc in h_cols
            for col in EXOG_COLS if _safe(col) in fc)
        print("Все future_exog нулевые — оставляем все")

    for col in good_exog_vars:
        if col not in selected_exog_ccf_order:
            continue
        hf = _has_flag(col)
        if hf in global_df_v2.columns and hf not in final_base:
            final_base.append(hf)

    valid_per_h: dict[int, list] = {}
    for h, cols in exog_future_cols_per_h.items():
        valid_per_h[h] = [fc for fc in cols
            if fc in global_df_v2.columns
            and any(_safe(col) in fc for col in good_exog_vars)
            and any(_safe(col) in fc for col in selected_exog_ccf_order)]


    final_exog = [col for col in selected_exog_ccf_order if col in good_exog_vars]
    dropped = len([f for f in base_raw if not f.startswith("has_") and f not in nonzero_overall])
    print(f"Базовых (с has_flags): {len(final_base)}")
    print(f"has_flags: {[f for f in final_base if f.startswith('has_')]}")
    print(f"Экзогенных: {len(final_exog)}: {final_exog}")
    print(f"Выброшено нулевых базовых: {dropped}")
    return final_base, valid_per_h, final_exog


def _make_subset_features(global_df_v2, base_features, exog_future_cols_per_h,exog_subset, architecture):
    if architecture == ARCH_NO_EXOG:
        base = [f for f in base_features if not f.startswith("has_") and "future_h" not in f]
        return base, {h: [] for h in exog_future_cols_per_h}

    sub_per_h = {
        h: [fc for fc in cols if any(_safe(c) in fc for c in exog_subset)]
        for h, cols in exog_future_cols_per_h.items()}
    if architecture == ARCH_NO_FLAGS:
        base = [f for f in base_features if not f.startswith("has_")]
        return base, sub_per_h

    base = [f for f in base_features if not f.startswith("has_") or any(_safe(c) in f for c in exog_subset)]
    for col in exog_subset:
        hf = _has_flag(col)
        if hf not in base and hf in global_df_v2.columns:
            base.append(hf)
    return base, sub_per_h


def _update_calendar(x_h, base_features, fd):
    for feat, val in [
        ("month", int(fd.month)),
        ("month_sin", float(np.sin(2*np.pi*fd.month/12))),
        ("month_cos", float(np.cos(2*np.pi*fd.month/12))),
        ("quarter", int(fd.quarter)),
        ("quarter_sin", float(np.sin(2*np.pi*fd.quarter/4))),
        ("quarter_cos", float(np.cos(2*np.pi*fd.quarter/4))),
        ("is_q4", int(fd.month >= 10)),
        ("is_summer", int(fd.month in [6, 7, 8])),
        ("covid", 0), ("post_covid", 0)]:
        if feat in base_features:
            x_h[feat] = val
    return x_h


def _cv_mape(
    global_df_v2, base_features, exog_per_h, selected_exog,
    last_value_per_series, global_last,
    cv_folds, model_fn, architecture, mask_prob=0.3,
    mode="no_exog", eval_horizons=None) -> float:
    if eval_horizons is None:
        eval_horizons = CV_HORIZONS

    mapes = []
    for train_mask, test_mask, _ in cv_folds:
        train_df = global_df_v2.loc[train_mask].copy()
        test_df = global_df_v2.loc[test_mask].copy()

        fc = DirectForecasterV2(
            model_fn=model_fn, name="cv",
            horizons=eval_horizons,
            base_features=base_features,
            exog_future_cols_per_h={h: [c for c in cols if c in train_df.columns]
                for h, cols in exog_per_h.items()},
            selected_exog=selected_exog,
            last_value_per_series=last_value_per_series,
            global_last=global_last,
            architecture=architecture,
            mask_prob=mask_prob,
            scale_features=False)
        fc.fit(train_df)

        for (cat, ch), grp in test_df.groupby(["_category", "_channel"]):
            tr_s = train_df[(train_df["_category"] == cat) & (train_df["_channel"] == ch)]
            if tr_s.empty:
                continue
            x_T = tr_s[base_features].fillna(0).iloc[[-1]].copy()
            for h_idx, fd_str in enumerate(sorted(grp["_date"].unique())):
                h = h_idx+1
                if h not in fc.models or h not in eval_horizons:
                    continue
                fact = grp[grp["_date"] == fd_str]
                if fact.empty:
                    continue
                y_true = float(np.expm1(fact[TARGET_COL].values[0]))
                x_h = _update_calendar(x_T.copy(), base_features, pd.Timestamp(fd_str))
                if "t" in base_features and "t" in x_T.columns:
                    new_t = float(x_T["t"].values[0] + h)
                    x_h["t"] = new_t
                    if "t_squared" in base_features:
                        x_h["t_squared"] = new_t ** 2

                ue = ({col: float(fact[col].values[0])
                       for col in selected_exog
                       if col in fact.columns
                       and not np.isnan(float(fact[col].values[0]))}
                      if mode == "full_exog" else None)

                y_pred = max(0.0, float(fc.predict(x_h, h, cat=cat, ch=ch, user_exog=ue)[0]))
                mape_h = abs(y_true - y_pred) / (abs(y_true) + 1e-10) * 100
                if not np.isnan(mape_h):
                    mapes.append(mape_h)

    return float(np.mean(mapes)) if mapes else np.inf


def step_experiment_A(
    global_df_v2, base_features, exog_future_cols_per_h,
    final_exog_vars, last_value_per_series, global_last,
    cv_folds, make_lgbm_fn, config_name="all_exog") -> list:

    if not final_exog_vars:
        return []

    n = len(final_exog_vars)
    print(f"Всего переменных для перебора: {n}")
    print(f"Порядок (по убыванию CCF): {final_exog_vars}")

    results = {}
    for k in range(1, n+1):
        subset = final_exog_vars[:k]
        set_name = f"top{k}"

        arch_base, arch_per_h = _make_subset_features(global_df_v2, base_features, exog_future_cols_per_h,
            subset, ARCH_WITH_FLAGS)
        mape = _cv_mape(
            global_df_v2, arch_base, arch_per_h, subset,
            last_value_per_series, global_last,
            cv_folds, make_lgbm_fn, architecture=ARCH_WITH_FLAGS,
            mask_prob=0.3, mode="no_exog")
        results[set_name] = {"exog": subset, "mape": mape}
        print(f"  top{k:<3}: {subset[-1]:<30} "
              f"(+1 к предыдущему), CV MAPE={mape:.2f}%")

    best_name = min(results, key=lambda k: (results[k]["mape"], len(results[k]["exog"])))
    best_exog = results[best_name]["exog"]

    for set_name, res in results.items():
        marker = "winner" if set_name == best_name else ""
        print(f"{set_name:<8} {res['exog'][-1]:<30} {res['mape']:>10.2f}%{marker}")

    print(f"\nПобедитель A [{config_name}]: {best_name} ({len(best_exog)} перем.)")
    print(f"{best_exog}")

    with mlflow.start_run(run_name=f"exp_A_{config_name}"):
        mlflow.set_tag("experiment", f"A_{config_name}")
        mlflow.set_tag("config_name", config_name)
        mlflow.set_tag("n_candidates", str(n))
        for set_name, res in results.items():
            mlflow.log_metric(f"cv_mape_{set_name}", round(res["mape"], 3))
        mlflow.log_param("winner", best_name)
        mlflow.log_param("best_exog", str(best_exog))
        mlflow.log_param("n_best_exog", len(best_exog))
    return best_exog


def step_experiment_D(
    global_df_v2, base_features, exog_future_cols_per_h,
    final_exog_vars, last_value_per_series, global_last,
    cv_folds, make_lgbm_fn, config_name="all_exog") -> str:

    if not final_exog_vars:
        return ARCH_NO_EXOG

    results = {}
    for arch in [ARCH_NO_EXOG, ARCH_NO_FLAGS, ARCH_WITH_FLAGS]:
        arch_base, arch_per_h = _make_subset_features(global_df_v2, base_features, exog_future_cols_per_h,final_exog_vars, arch)
        mape = _cv_mape(
            global_df_v2, arch_base, arch_per_h, final_exog_vars,
            last_value_per_series, global_last,
            cv_folds, make_lgbm_fn, architecture=arch,
            mask_prob=0.3, mode="no_exog")
        results[arch] = mape
        label = {
            ARCH_NO_EXOG:"Без экзогенных:",
            ARCH_NO_FLAGS:"С экзог, без флагов:",
            ARCH_WITH_FLAGS: "С экзог и флагами:"}[arch]
        print(f"{label} CV no_exog MAPE = {mape:.2f}%")
    best_arch = min(results, key=results.get)
    print(f"Победитель D [{config_name}]:{best_arch} MAPE={results[best_arch]:.2f}%")

    with mlflow.start_run(run_name=f"exp_D_{config_name}"):
        mlflow.set_tag("experiment", f"D_{config_name}")
        mlflow.set_tag("config_name", config_name)
        for arch, mape in results.items():
            mlflow.log_metric(f"cv_mape_{arch}", round(mape, 3))
        mlflow.log_param("winner_architecture", best_arch)
        mlflow.log_param("exog_vars_tested", str(final_exog_vars))
    return best_arch


def step_experiment_B(
    global_df_v2, base_features, exog_future_cols_per_h,
    best_exog, last_value_per_series, global_last,
    cv_folds, make_lgbm_fn, best_arch, config_name="all_exog") -> float:

    if best_arch != ARCH_WITH_FLAGS:
        return 0.3

    arch_base, arch_per_h = _make_subset_features(global_df_v2, base_features, exog_future_cols_per_h, best_exog, best_arch)
    results = {}
    for mp in MASK_PROB_CANDIDATES:
        m_no = _cv_mape(global_df_v2, arch_base, arch_per_h, best_exog,
                           last_value_per_series, global_last, cv_folds,
                           make_lgbm_fn, architecture=best_arch,
                           mask_prob=mp, mode="no_exog")
        m_full = _cv_mape(global_df_v2, arch_base, arch_per_h, best_exog,
                           last_value_per_series, global_last, cv_folds,
                           make_lgbm_fn, architecture=best_arch,
                           mask_prob=mp, mode="full_exog")
        results[mp] = {"no_exog": m_no, "full_exog": m_full}
        print(f"p={mp}: no_exog={m_no:.2f}%  full_exog={m_full:.2f}%")

    best_mp = float(min(results, key=lambda p: results[p]["no_exog"]))
    print(f"\nПобедитель B [{config_name}]: mask_prob={best_mp}")

    with mlflow.start_run(run_name=f"exp_B_{config_name}"):
        mlflow.set_tag("experiment", f"B_{config_name}")
        mlflow.set_tag("config_name", config_name)
        for mp, res in results.items():
            mlflow.log_metric(f"no_exog_p{int(mp*100)}", round(res["no_exog"], 3))
            mlflow.log_metric(f"full_exog_p{int(mp*100)}", round(res["full_exog"], 3))
        mlflow.log_param("best_mask_prob", best_mp)
    return best_mp


def step_experiment_C(
    global_df_v2, base_features, exog_future_cols_per_h,
    best_exog, last_value_per_series, global_last,
    cv_folds, make_lgbm_fn, best_arch, best_mask_prob, config_name="all_exog") -> bool:

    exog_to_use = best_exog if best_arch != ARCH_NO_EXOG else []
    arch_base, arch_per_h = _make_subset_features(global_df_v2, base_features, exog_future_cols_per_h, exog_to_use, best_arch)
    unified_mape = _cv_mape(
        global_df_v2, arch_base, arch_per_h, exog_to_use,
        last_value_per_series, global_last, cv_folds, make_lgbm_fn,
        architecture=best_arch, mask_prob=best_mask_prob, mode="no_exog")
    print(f"Единая модель: {unified_mape:.2f}%")

    lgbm_base = {**_load_tuned_params().get("lgbm", LGBM_DEFAULT_PARAMS),"verbose": -1, "random_state": RANDOM_STATE}
    short_p = {**lgbm_base, "reg_lambda":0.3}
    long_p = {**lgbm_base, "reg_lambda":3.0}
    short_hs = [h for h in CV_HORIZONS if h<=3]
    long_hs = [h for h in CV_HORIZONS if h > 3]

    split_mapes = []
    for train_mask, test_mask, _ in cv_folds:
        train_df = global_df_v2.loc[train_mask].copy()
        test_df = global_df_v2.loc[test_mask].copy()
        kw = dict(base_features=arch_base, selected_exog=exog_to_use,
                  last_value_per_series=last_value_per_series, global_last=global_last,
                  architecture=best_arch, mask_prob=best_mask_prob)
        fcs = (DirectForecasterV2(
            model_fn=lambda: lgb.LGBMRegressor(**short_p), name="short",
            horizons=short_hs,
            exog_future_cols_per_h={h: arch_per_h.get(h, []) for h in short_hs},
            **kw) if short_hs else None)
        fcl = (DirectForecasterV2(
            model_fn=lambda: lgb.LGBMRegressor(**long_p), name="long",
            horizons=long_hs,
            exog_future_cols_per_h={h: arch_per_h.get(h, []) for h in long_hs},
            **kw) if long_hs else None)
        if fcs: fcs.fit(train_df)
        if fcl: fcl.fit(train_df)
        for (cat, ch), grp in test_df.groupby(["_category", "_channel"]):
            tr_s = train_df[(train_df["_category"] == cat) & (train_df["_channel"] == ch)]
            if tr_s.empty: continue
            x_T = tr_s[arch_base].fillna(0).iloc[[-1]].copy()
            for h_idx, fd_str in enumerate(sorted(grp["_date"].unique())):
                h = h_idx+1
                if h not in CV_HORIZONS: continue
                fact = grp[grp["_date"] == fd_str]
                if fact.empty: continue
                y_true = float(np.expm1(fact[TARGET_COL].values[0]))
                x_h = _update_calendar(x_T.copy(), arch_base, pd.Timestamp(fd_str))
                fc_use = (fcs if h in short_hs and fcs else fcl)
                if fc_use is None or h not in fc_use.models: continue
                y_pred = max(0.0, float(fc_use.predict(x_h, h, cat=cat, ch=ch)[0]))
                mape_h = abs(y_true - y_pred)/(abs(y_true)+1e-10)*100
                if not np.isnan(mape_h): split_mapes.append(mape_h)

    split_mape = float(np.mean(split_mapes)) if split_mapes else np.inf
    use_split = split_mape < unified_mape
    print(f"Горизонт-split: {split_mape:.2f}%   delta={split_mape-unified_mape:+.2f}%")

    with mlflow.start_run(run_name=f"exp_C_{config_name}"):
        mlflow.set_tag("experiment", f"C_{config_name}")
        mlflow.set_tag("config_name", config_name)
        mlflow.log_metric("unified_cv_mape", round(unified_mape, 3))
        mlflow.log_metric("split_cv_mape", round(split_mape, 3))
        mlflow.log_param("winner", "split" if use_split else "unified")
    return use_split


def step_tune_hyperparams_v2(
    global_df_v2, base_features, exog_future_cols_per_h,
    best_exog, last_value_per_series, global_last,
    cv_folds, best_arch, best_mask_prob, config_name="all_exog") -> dict:

    exog_to_use = best_exog if best_arch != ARCH_NO_EXOG else []
    arch_base, arch_per_h = _make_subset_features(
        global_df_v2, base_features, exog_future_cols_per_h, exog_to_use, best_arch)
    base_params = _load_tuned_params()
    lgbm_base = {**base_params["lgbm"], "verbose": -1, "random_state": RANDOM_STATE}
    xgb_base = {**base_params["xgb"], "verbosity": 0, "random_state": RANDOM_STATE}
    tuned = {}

    print(f"[LightGBM]")
    def lgbm_obj(trial):
        md = lgbm_base.get("max_depth", 5)
        p = {**lgbm_base,
              "num_leaves":        trial.suggest_int("num_leaves", 15, min(127, 2**md-1)),
              "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.5, 1.0),
              "min_child_samples": trial.suggest_int("min_child_samples", 5, 50)}
        return _cv_mape(global_df_v2, arch_base, arch_per_h, exog_to_use,
                         last_value_per_series, global_last, cv_folds,
                         lambda: lgb.LGBMRegressor(**p),
                         architecture=best_arch, mask_prob=best_mask_prob, mode="no_exog")

    lgbm_study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    lgbm_study.optimize(lgbm_obj, n_trials=OPTUNA_N_TRIALS_V2, show_progress_bar=True)
    tuned["lgbm"] = {**lgbm_base, **lgbm_study.best_params}
    print(f"Best MAPE: {lgbm_study.best_value:.2f}%  params: {lgbm_study.best_params}")

    print(f"[XGBoost]")
    def xgb_obj(trial):
        p = {**xgb_base,
             "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
             "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
             "max_depth":        trial.suggest_int("max_depth", 3, 7)}
        return _cv_mape(global_df_v2, arch_base, arch_per_h, exog_to_use,
                         last_value_per_series, global_last, cv_folds,
                         lambda: xgb_module.XGBRegressor(**p),
                         architecture=best_arch, mask_prob=best_mask_prob, mode="no_exog")

    xgb_study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    xgb_study.optimize(xgb_obj, n_trials=OPTUNA_N_TRIALS_V2, show_progress_bar=True)
    tuned["xgb"] = {**xgb_base, **xgb_study.best_params}
    print(f"  Best MAPE: {xgb_study.best_value:.2f}%  params: {xgb_study.best_params}")

    print(f"[RandomForest]")
    rf_base = {**_load_tuned_params().get("rf", RF_DEFAULT_PARAMS),
               "n_jobs": -1, "random_state": RANDOM_STATE}

    def rf_obj(trial):
        p = {**rf_base,
             "n_estimators":    trial.suggest_int("n_estimators", 100, 400),
             "max_depth":       trial.suggest_int("max_depth", 3, 8),
             "min_samples_leaf": trial.suggest_int("min_samples_leaf", 3, 30),
             "max_features":    trial.suggest_categorical(
                 "max_features", ["sqrt", "log2", 0.5]),
             "n_jobs": -1, "random_state": RANDOM_STATE}
        return _cv_mape(global_df_v2, arch_base, arch_per_h, exog_to_use,
                         last_value_per_series, global_last, cv_folds,
                         lambda: RandomForestRegressor(**p),
                         architecture=best_arch, mask_prob=best_mask_prob, mode="no_exog")

    rf_study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    rf_study.optimize(rf_obj, n_trials=OPTUNA_N_TRIALS_RF, show_progress_bar=True)
    tuned["rf"] = {**rf_base, **rf_study.best_params}
    print(f"Best MAPE: {rf_study.best_value:.2f}%  params: {rf_study.best_params}")

    for model_nm, study in [("lgbm", lgbm_study), ("xgb", xgb_study), ("rf", rf_study)]:
        with mlflow.start_run(run_name=f"optuna_{model_nm}_{config_name}"):
            mlflow.set_tag("experiment", f"optuna_{model_nm}_{config_name}")
            mlflow.set_tag("config_name", config_name)
            mlflow.log_params(study.best_params)
            mlflow.log_metric("best_cv_mape", study.best_value)

    params_path = ML_MODELS_DIR / f"ml_v2_{config_name}_tuned_params.json"
    with open(params_path, "w") as f:
        json.dump(tuned, f, ensure_ascii=False, indent=2)
    return tuned


def step_train(
    global_df_v2, base_features, exog_future_cols_per_h,
    best_exog, last_value_per_series, global_last,
    tuned_params, best_arch, best_mask_prob, horizons, config_name="all_exog") -> dict:

    exog_to_use = best_exog if best_arch != ARCH_NO_EXOG else []
    arch_base, arch_per_h = _make_subset_features(global_df_v2, base_features, exog_future_cols_per_h, exog_to_use, best_arch)

    lgbm_p = {**tuned_params.get("lgbm", LGBM_DEFAULT_PARAMS),"verbose": -1, "random_state": RANDOM_STATE}
    xgb_p = {**tuned_params.get("xgb", XGB_DEFAULT_PARAMS),"verbosity": 0, "random_state": RANDOM_STATE}
    rf_p = {**tuned_params.get("rf", RF_DEFAULT_PARAMS),"n_jobs": -1, "random_state": RANDOM_STATE}
    en_p = {**EN_DEFAULT_PARAMS}

    factories = {"LightGBM":(ModelFactoryV2("lgbm", lgbm_p), False),
        "XGBoost":(ModelFactoryV2("xgb", xgb_p), False),
        "RandomForest":(ModelFactoryV2("rf", rf_p), False),
        "ElasticNet":(ModelFactoryV2("elasticnet", en_p), True)}

    dates = sorted(global_df_v2["_date"].unique())
    train_df = global_df_v2[global_df_v2["_date"].isin(set(dates[:-TEST_SIZE]))].copy()
    full_df = global_df_v2.copy()
    trained = {}
    for model_name, (model_fn, scale) in factories.items():
        print(f"\n  [{model_name}]  arch={best_arch}")
        kw = dict(
            horizons=horizons, base_features=arch_base,
            exog_future_cols_per_h=arch_per_h, selected_exog=exog_to_use,
            last_value_per_series=last_value_per_series, global_last=global_last,
            architecture=best_arch, mask_prob=best_mask_prob, scale_features=scale)
        print(f"eval ({len(train_df)} строк)", end=" ", flush=True)
        fc_eval = DirectForecasterV2(model_fn=model_fn, name=f"{model_name}_eval", **kw)
        fc_eval.fit(train_df)
        print("OK")

        print(f"full ({len(full_df)} строк)", end=" ", flush=True)
        fc_full = DirectForecasterV2(model_fn=model_fn, name=f"{model_name}_full", **kw)
        fc_full.fit(full_df)
        print("OK")
        trained[model_name] = {"eval": fc_eval, "full": fc_full, "base": arch_base}
    return trained


def step_evaluate_h12(trained, global_df_v2, best_exog, best_arch) -> pd.DataFrame:

    dates = sorted(global_df_v2["_date"].unique())
    test_dates = set(dates[-TEST_SIZE:])
    train_dates = set(dates[:-TEST_SIZE])
    test_df = global_df_v2[global_df_v2["_date"].isin(test_dates)].copy()
    train_df = global_df_v2[global_df_v2["_date"].isin(train_dates)].copy()
    partial_col = best_exog[0] if best_exog and best_arch != ARCH_NO_EXOG else None
    all_rows = []
    for model_name, bundle in trained.items():
        fc = bundle["eval"]
        base_features = bundle["base"]
        for (cat, ch), grp_te in test_df.groupby(["_category", "_channel"]):
            tr_s = train_df[(train_df["_category"] == cat) & (train_df["_channel"] == ch)]
            if tr_s.empty: continue
            x_T = tr_s[base_features].fillna(0).iloc[[-1]].copy()
            te_dates = sorted(grp_te["_date"].unique())
            preds = {"no_exog": [], "partial_exog": [], "full_exog": []}
            y_trues = []

            for h, fd_str in enumerate(te_dates, start=1):
                fact = grp_te[grp_te["_date"] == fd_str]
                if fact.empty or h not in fc.models: continue
                y_trues.append(float(np.expm1(fact[TARGET_COL].values[0])))
                x_h = _update_calendar(x_T.copy(), base_features, pd.Timestamp(fd_str))
                if "t" in base_features and "t" in x_T.columns:
                    new_t = float(x_T["t"].values[0] + h)
                    x_h["t"] = new_t
                    if "t_squared" in base_features:
                        x_h["t_squared"] = new_t ** 2
                real_exog = ({
                    col: float(fact[col].values[0])
                    for col in best_exog
                    if col in fact.columns and not np.isnan(float(fact[col].values[0]))} if best_arch != ARCH_NO_EXOG else {})

                preds["no_exog"].append(max(0.0, float(fc.predict(x_h, h, cat=cat, ch=ch, user_exog=None)[0])))
                p_ue = ({partial_col: real_exog[partial_col]} if partial_col and partial_col in real_exog else None)
                preds["partial_exog"].append(max(0.0, float(fc.predict(x_h, h, cat=cat, ch=ch, user_exog=p_ue)[0])))
                preds["full_exog"].append(max(0.0, float(fc.predict(x_h, h, cat=cat, ch=ch,user_exog=real_exog if real_exog else None)[0])))

            if not y_trues: continue
            y_arr = np.array(y_trues)
            for mode, y_list in preds.items():
                if y_list:
                    m = compute_metrics(y_arr, np.array(y_list))
                    all_rows.append({"category": cat, "channel": ch,"model": model_name, "mode": mode, **m})
        print("OK")

    out = pd.DataFrame(all_rows)
    pivot = out.groupby(["model", "mode"])["mape"].median().unstack().round(2)
    for c in ["no_exog", "partial_exog", "full_exog"]:
        if c not in pivot.columns: pivot[c] = np.nan
    print(pivot[["no_exog", "partial_exog", "full_exog"]].to_string())
    return out


def step_select_best_model(test_metrics: pd.DataFrame) -> str:
    summary = (
        test_metrics[test_metrics["mode"]=="no_exog"]
        .groupby("model")["mape"]
        .agg(["median", "mean", "std"])
        .round(2)
        .sort_values("median"))
    print(summary.to_string())
    best = summary.index[0]
    print(f"Победитель: {best}{summary.loc[best,'median']:.2f}")
    return best


def step_save(
    trained, best_model, best_exog, best_arch,
    exog_future_cols_per_h, has_flag_cols,
    last_value_per_series, global_last,
    lag_ranges, cat_enc, ch_enc, best_mask_prob, tuned_params,
    config_name="all_exog"):
    for model_name, bundle in trained.items():
        eval_name, full_name = PKL_NAMES[model_name]
        with open(ML_MODELS_DIR / eval_name, "wb") as f:
            pickle.dump(bundle["eval"], f)
        with open(ML_MODELS_DIR / full_name, "wb") as f:
            pickle.dump(bundle["full"], f)

    _, winner_pkl = PKL_NAMES[best_model]
    base_features = trained[best_model]["base"]
    lv_json = {f"{k[0]}|{k[1]}": v for k, v in last_value_per_series.items()}
    meta = {
        "version": f"v2_{config_name}",
        "config_name": config_name,
        "best_model": best_model,
        "winner_pkl_full": str(ML_MODELS_DIR / winner_pkl),
        "architecture": best_arch,
        "base_features": base_features,
        "selected_exog": best_exog,
        "has_flag_cols": has_flag_cols,
        "last_value_per_series":  lv_json,
        "global_last": global_last,
        "exog_cols": EXOG_COLS,
        "mask_prob": best_mask_prob,
        "ccf_lag_ranges": lag_ranges,
        "cat_encoder": cat_enc,
        "ch_encoder": ch_enc,
        "horizon": HORIZON,
        "test_size": TEST_SIZE,
        "exog_future_cols_per_h": {str(h): v for h, v in exog_future_cols_per_h.items()},
        "pkl_names": PKL_NAMES,
        "tuned_params": tuned_params,
        "strategy": "Direct Multi-Step",
        "imputation": "last_value_per_series",
        "inference_no_exog": "user_exog=None:last_value+has_flag=0",
        "inference_with_exog":  "user_exog={col:val}: real + has_flag=1",
        "full_exog_note": "full_exog = upper bound only"}
    meta_path = ML_MODELS_DIR / f"ml_v2_{config_name}_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def _get_part1_mape() -> dict:
    try:
        client = mlflow.tracking.MlflowClient()
        for exp_name in [MLFLOW_EXPERIMENT, "mars-forecasting", "mars_sarima"]:
            exp = client.get_experiment_by_name(exp_name)
            if exp is None: continue
            runs = client.search_runs(experiment_ids=[exp.experiment_id], max_results=2000)
            result = {}
            for run in runs:
                parts = run.data.tags.get("mlflow.runName", "").split("|")
                if len(parts) < 3: continue
                mape = run.data.metrics.get("best_mape")
                if mape: result[(parts[1].strip(), parts[2].strip())] = float(mape)
            if result: return result
    except Exception: pass
    return {}


def _get_baseline_ml_mape() -> dict:
    try:
        client = mlflow.tracking.MlflowClient()
        exp = client.get_experiment_by_name("mars_model_selection")
        if exp is None: return {}
        runs = client.search_runs(experiment_ids=[exp.experiment_id], max_results=2000)
        return {
            (r.data.tags.get("category"), r.data.tags.get("channel")):
            float(r.data.metrics["ml_mape"])
            for r in runs
            if r.data.tags.get("category") and "ml_mape" in r.data.metrics}
    except Exception: return {}


def _row_mape(df, cat, ch, model, mode):
    sub = df[(df["category"] == cat) & (df["channel"] == ch) & (df["model"] == model) & (df["mode"] == mode)]
    return float(sub["mape"].values[0]) if len(sub) > 0 else np.nan


def step_log_mlflow(
    test_metrics, trained, best_model, best_exog,
    best_arch, best_mask_prob,
    config_name: str = "all_exog",
):

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_V2)
    base_features = trained[best_model]["base"]
    _, winner_pkl = PKL_NAMES[best_model]
    winner_path = str(ML_MODELS_DIR / winner_pkl)
    best_mape = float(test_metrics[(test_metrics["model"] == best_model) & (test_metrics["mode"] == "no_exog")]["mape"].median())

    for model_name in trained.keys():
        def _med(mode, mn=model_name):
            sub = test_metrics[(test_metrics["model"] == mn) & (test_metrics["mode"] == mode)]
            return float(sub["mape"].median()) if len(sub) > 0 else np.nan

        with mlflow.start_run(run_name=f"{model_name.lower()}_{config_name}_final"):
            mlflow.set_tag("model_type", model_name)
            mlflow.set_tag("architecture", best_arch)
            mlflow.set_tag("config_name", config_name)
            mlflow.set_tag("best_model", str(model_name == best_model).lower())
            mlflow.log_params({
                "n_base_features": len(base_features),
                "selected_exog": str(best_exog),
                "n_exog": len(best_exog),
                "mask_prob": best_mask_prob,
                "architecture": best_arch,
                "config_name":config_name})
            m_dict = {}
            for mode in ["no_exog", "partial_exog", "full_exog"]:
                v = _med(mode)
                if not np.isnan(v):
                    m_dict[f"test_mape_h12_{mode}"] = round(v, 3)
            if "test_mape_h12_no_exog" in m_dict:
                m_dict["test_mape_h12_median"] = m_dict["test_mape_h12_no_exog"]
            mlflow.log_metrics(m_dict)
            csv_path = ML_RESULTS_DIR / f"test_metrics_v2_{config_name}.csv"
            meta_path = ML_MODELS_DIR / f"ml_v2_{config_name}_metadata.json"
            for path in [csv_path, meta_path]:
                if path.exists(): mlflow.log_artifact(str(path))

    mlflow.set_experiment(MLFLOW_EXPERIMENT_V2_COMPARE)
    part1_map = _get_part1_mape()
    baseline_map = _get_baseline_ml_mape()
    pairs = test_metrics[["category", "channel"]].drop_duplicates().values.tolist()

    for cat, ch in pairs:
        mape_no = _row_mape(test_metrics, cat, ch, best_model, "no_exog")
        mape_part = _row_mape(test_metrics,cat, ch, best_model, "partial_exog")
        mape_full = _row_mape(test_metrics, cat, ch, best_model, "full_exog")
        p1 = part1_map.get((cat, ch))
        bl = baseline_map.get((cat, ch))
        cands = {}
        if not np.isnan(mape_no): cands["v2_no_exog"] = mape_no
        if p1 is not None:cands["part1"] = p1
        if bl is not None: cands["ml_baseline"] = bl
        winner_overall = min(cands, key=cands.get) if cands else "unknown"
        with mlflow.start_run(run_name=f"{cat}|{ch}|{config_name}"):
            mlflow.set_tag("category", cat)
            mlflow.set_tag("channel", ch)
            mlflow.set_tag("config_name", config_name)
            mlflow.set_tag("winner_v2_model", best_model)
            mlflow.set_tag("winner_overall", winner_overall)
            mlflow.set_tag("ml_v2_pkl_full", winner_path)
            mlflow.set_tag("architecture", best_arch)
            mlflow.set_tag("selected_exog", str(best_exog))

            m = {}
            if not np.isnan(mape_no): m["ml_v2_mape_no_exog"] = round(mape_no, 3)
            if not np.isnan(mape_part): m["ml_v2_mape_partial_exog"] = round(mape_part, 3)
            if not np.isnan(mape_full): m["ml_v2_mape_full_exog"] = round(mape_full, 3)
            if p1 is not None and not np.isnan(mape_no):
                m["part1_mape"] = round(p1, 3)
                m["delta_vs_part1"] = round(mape_no-p1, 3)
            if bl is not None and not np.isnan(mape_no):
                m["ml_baseline_mape"] = round(bl, 3)
                m["delta_vs_baseline"] = round(mape_no - bl, 3)
            if m: mlflow.log_metrics(m)

    v2_wins = sum(1 for cat, ch in pairs
        if (p1 := part1_map.get((cat, ch))) is not None
        and not np.isnan(_row_mape(test_metrics, cat, ch, best_model, "no_exog"))
        and _row_mape(test_metrics, cat, ch, best_model, "no_exog") < p1)

    with mlflow.start_run(run_name=f"_summary_{config_name}"):
        mlflow.set_tag("winner_model", best_model)
        mlflow.set_tag("architecture", best_arch)
        mlflow.set_tag("config_name", config_name)
        mlflow.log_metrics({
            "total_series":len(pairs),
            "best_mape_no_exog": round(best_mape, 3),
            "v2_wins_vs_part1": v2_wins})

        for mode in ["no_exog", "partial_exog", "full_exog"]:
            sub = test_metrics[test_metrics["mode"] == mode]
            for mn, mp in sub.groupby("model")["mape"].median().items():
                safe_mn = mn.lower().replace(" ", "_")
                mlflow.log_metric(f"median_{safe_mn}_{mode}", round(float(mp), 3))


def _run_one_config(
    config_name: str,
    exog_subset: list[str],
    global_df_v2: pd.DataFrame,
    feature_cols: list,
    exog_future_cols_per_h: dict,
    all_future_cols: list,
    has_flag_cols: list,
    last_value_per_series: dict,
    global_last: dict,
    cv_folds: list,
    lag_ranges: dict,
    cat_enc: dict,
    ch_enc: dict,
    horizons: list,
    selected_exog_ccf_order: list) -> tuple[pd.DataFrame, str, str]:

    if not exog_subset:
        print("Пустой exog_subset")

    make_lgbm_base = lambda: lgb.LGBMRegressor(**{**_load_tuned_params().get("lgbm", LGBM_DEFAULT_PARAMS),
           "verbose": -1, "random_state": RANDOM_STATE})
    mlflow.set_experiment(MLFLOW_EXPERIMENT_V2)

    base_features, exog_per_h, final_exog = step_select_features_v2(
        global_df_v2, feature_cols, exog_future_cols_per_h,
        all_future_cols, has_flag_cols,
        selected_exog_ccf_order=exog_subset)

    best_exog = step_experiment_A(
        global_df_v2, base_features, exog_per_h,
        final_exog, last_value_per_series, global_last,
        cv_folds, make_lgbm_base, config_name=config_name)

    best_arch = step_experiment_D(
        global_df_v2, base_features, exog_per_h,
        best_exog if best_exog else final_exog,
        last_value_per_series, global_last,
        cv_folds, make_lgbm_base, config_name=config_name)
    if best_arch == ARCH_NO_EXOG:
        best_exog = []


    best_mask_prob = step_experiment_B(
        global_df_v2, base_features, exog_per_h,
        best_exog, last_value_per_series, global_last,
        cv_folds, make_lgbm_base, best_arch, config_name=config_name)

    step_experiment_C(
        global_df_v2, base_features, exog_per_h,
        best_exog, last_value_per_series, global_last,
        cv_folds, make_lgbm_base, best_arch, best_mask_prob, config_name=config_name)


    tuned_params = step_tune_hyperparams_v2(
        global_df_v2, base_features, exog_per_h,
        best_exog, last_value_per_series, global_last,
        cv_folds, best_arch, best_mask_prob, config_name=config_name)

    trained = step_train(
        global_df_v2, base_features, exog_per_h,
        best_exog, last_value_per_series, global_last,
        tuned_params, best_arch, best_mask_prob, horizons, config_name=config_name)

    test_metrics_raw = step_evaluate_h12(trained, global_df_v2, best_exog, best_arch)

    test_metrics = test_metrics_raw.copy()
    test_metrics["model"] = test_metrics["model"] + f"_{config_name}"
    best_model_raw = step_select_best_model(test_metrics_raw)
    best_model = f"{best_model_raw}_{config_name}"
    orig_pkl = dict(PKL_NAMES)
    for mn in list(PKL_NAMES.keys()):
        e, f_ = PKL_NAMES[mn]
        PKL_NAMES[mn] = (e.replace("ml_v2_", f"ml_v2_{config_name}_"),f_.replace("ml_v2_", f"ml_v2_{config_name}_"))

    step_save(
        trained, best_model_raw, best_exog, best_arch,
        exog_per_h, has_flag_cols,
        last_value_per_series, global_last,
        lag_ranges, cat_enc, ch_enc, best_mask_prob, tuned_params,
        config_name=config_name)
    PKL_NAMES.update(orig_pkl)
    csv_path = ML_RESULTS_DIR / f"test_metrics_v2_{config_name}.csv"
    test_metrics_raw.to_csv(csv_path, index=False)

    step_log_mlflow(test_metrics_raw, trained, best_model_raw,
        best_exog, best_arch, best_mask_prob,
        config_name=config_name)

    return test_metrics, best_model, best_arch, trained, best_exog, best_mask_prob, tuned_params


def print_full_comparison(
    metrics_all: pd.DataFrame,
    metrics_vif: pd.DataFrame,
    best_all: str,
    best_vif: str):
    part1_map = _get_part1_mape()
    baseline_map = _get_baseline_ml_mape()
    rows = []
    for metrics, best in [(metrics_all, best_all), (metrics_vif, best_vif)]:
        for model_name in sorted(metrics["model"].unique()):
            for mode in ["no_exog", "partial_exog", "full_exog"]:
                sub = metrics[(metrics["model"] == model_name) & (metrics["mode"] == mode)]
                mape = sub["mape"].median() if len(sub) > 0 else np.nan
                note = ""
                if mode == "no_exog" and model_name == best:
                    note = "лучший"
                elif mode == "full_exog":
                    note = " (upper bound)"
                rows.append({
                    "Модель":f"{model_name} [{mode}]{note}",
                    "Медиана MAPE": round(float(mape), 2)})

    if part1_map:
        rows.append({
            "Модель": "Эконометрика SARIMA/SARIMAX",
            "Медиана MAPE, %": round(np.median(list(part1_map.values())), 2)})
    if baseline_map:
        rows.append({"Модель": "ML baseline (без future exog)",
            "Медиана MAPE, %": round(np.median(list(baseline_map.values())), 2)})

    df = pd.DataFrame(rows).sort_values("Медиана MAPE, %")
    print(df.to_string(index=False))

    out = ML_RESULTS_DIR / "comparison_v2_full.csv"
    df.to_csv(out, index=False)
    print(f"\n{out}")

    no_exog_rows = df[
        df["Модель"].str.contains(r"\[no_exog\]", regex=True) |
        df["Модель"].str.contains("SARIMA", regex=False) |
        df["Модель"].str.contains("baseline", regex=False)]
    print(no_exog_rows.sort_values("Медиана MAPE, %").to_string(index=False))


def step_pin_production_model(
    metrics_vif: pd.DataFrame,
    trained_vif: dict,
    best_exog_vif: list,
    best_arch_vif: str,
    exog_future_cols_per_h: dict,
    has_flag_cols: list,
    last_value_per_series: dict,
    global_last: dict,
    lag_ranges: dict,
    cat_enc: dict,
    ch_enc: dict,
    best_mask_prob_vif: float,
    tuned_params_vif: dict):

    if PROD_MODEL not in trained_vif:
        print(f"{PROD_MODEL} не найден в trained_vif. Пропускаем.")
        return

    bundle = trained_vif[PROD_MODEL]
    prod_eval_path = ML_MODELS_DIR / "ml_v2_prod_eval.pkl"
    prod_full_path = ML_MODELS_DIR / "ml_v2_prod_full.pkl"

    with open(prod_eval_path, "wb") as f:
        pickle.dump(bundle["eval"], f)
    with open(prod_full_path, "wb") as f:
        pickle.dump(bundle["full"], f)

    meta_path = ML_MODELS_DIR / f"ml_v2_{PROD_CONFIG}_metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
    else:
        meta = {}

    meta["best_model"] = PROD_MODEL
    meta["winner_pkl_full"] = str(prod_full_path)
    meta["winner_pkl_eval"] = str(prod_eval_path)
    meta["prod_pinned"] = True
    meta["prod_model"] = PROD_MODEL
    meta["prod_config"] = PROD_CONFIG
    meta["selected_exog"] = best_exog_vif
    meta["architecture"] = best_arch_vif
    meta["base_features"] = bundle["base"]

    with open(meta_path, "w") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    sub = metrics_vif[
        (metrics_vif["model"] == PROD_MODEL) & (metrics_vif["mode"] == "no_exog")]
    prod_mape = float(sub["mape"].median()) if len(sub) > 0 else None
    if prod_mape is not None:
        print(f"MAPE (no_exog, медиана): {prod_mape:.2f}%")
    mlflow.set_experiment(MLFLOW_EXPERIMENT_V2)
    with mlflow.start_run(run_name=f"production_{PROD_MODEL}_{PROD_CONFIG}"):
        mlflow.set_tag("is_production", "true")
        mlflow.set_tag("prod_model", PROD_MODEL)
        mlflow.set_tag("prod_config", PROD_CONFIG)
        mlflow.set_tag("prod_pkl_full", str(prod_full_path))
        mlflow.set_tag("architecture", best_arch_vif)
        mlflow.set_tag("selected_exog", str(best_exog_vif))
        if prod_mape is not None:
            mlflow.log_metric("prod_mape_no_exog", round(prod_mape, 3))
        mlflow.log_artifact(str(meta_path))

    print(f"Продакшн-модель зафиксирована: {PROD_MODEL} / {PROD_CONFIG}")
    print(f"API будет использовать: {prod_full_path.name}")


def main():
    horizons = list(range(1, HORIZON + 1))
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    (df_raw, series_dict,
     global_df, feature_cols,
     cat_enc, ch_enc, lag_ranges) = step_load()
    selected_exog = step_ccf_exog_selection(series_dict, global_df)
    vif_exog = step_vif_exog_selection(series_dict, global_df, selected_exog)
    (global_df_v2, exog_future_cols_per_h, all_future_cols,
     has_flag_cols, last_value_per_series, global_last) = \
        step_add_future_exog(global_df, series_dict, selected_exog, horizons)
    cv_folds = make_cv_folds(global_df_v2)
    step_leakage_check(global_df_v2, exog_future_cols_per_h, selected_exog)
    common = dict(
        global_df_v2=global_df_v2,
        feature_cols=feature_cols,
        exog_future_cols_per_h=exog_future_cols_per_h,
        all_future_cols=all_future_cols,
        has_flag_cols=has_flag_cols,
        last_value_per_series=last_value_per_series,
        global_last=global_last,
        cv_folds=cv_folds,
        lag_ranges=lag_ranges,
        cat_enc=cat_enc,
        ch_enc=ch_enc,
        horizons=horizons)

    (metrics_all, best_all, arch_all,
     _trained_all, _exog_all, _mp_all, _tp_all) = _run_one_config(
        config_name="all_exog",
        exog_subset=selected_exog,
        selected_exog_ccf_order=selected_exog,
        **common)

    (metrics_vif, best_vif, arch_vif,
     trained_vif, best_exog_vif, best_mask_prob_vif, tuned_params_vif) = _run_one_config(
        config_name="vif_exog",
        exog_subset=vif_exog,
        selected_exog_ccf_order=vif_exog,
        **common)

    step_pin_production_model(
        metrics_vif=metrics_vif,
        trained_vif=trained_vif,
        best_exog_vif=best_exog_vif,
        best_arch_vif=arch_vif,
        exog_future_cols_per_h=exog_future_cols_per_h,
        has_flag_cols=has_flag_cols,
        last_value_per_series=last_value_per_series,
        global_last=global_last,
        lag_ranges=lag_ranges,
        cat_enc=cat_enc,
        ch_enc=ch_enc,
        best_mask_prob_vif=best_mask_prob_vif,
        tuned_params_vif=tuned_params_vif)

    print_full_comparison(metrics_all, metrics_vif, best_all, best_vif)

    all_metrics = pd.concat([metrics_all, metrics_vif], ignore_index=True)
    all_metrics.to_csv(ML_RESULTS_DIR / "test_metrics_v2_both.csv", index=False)

    print(f"MLflow: mlflow ui --backend-store-uri {MLFLOW_TRACKING_URI}")

    return {
        "metrics_all":   metrics_all,
        "best_all":      best_all,
        "metrics_vif":   metrics_vif,
        "best_vif":      best_vif,
        "selected_exog": selected_exog,
        "vif_exog":      vif_exog,
        "series_dict":   series_dict}


if __name__ == "__main__":
    main()