import random
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import acf, pacf
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
matplotlib.use("Agg")
from statsmodels.tsa.stattools import ccf

from src.config.settings import (
    TARGET_COL, DATE_COL, CATEGORY_COL, CHANNEL_COL,
    EXOG_COLS, SEASONAL_PERIOD, RANDOM_STATE,
    TEST_SIZE, N_FOLDS, HORIZON,
    COVID_START, COVID_END,
)

from src.config.settings import TARGET_COL, SEASONAL_PERIOD, RANDOM_STATE

random.seed(RANDOM_STATE)


def get_series_stats(series_dict: dict) -> list[dict]:
    rows = []
    for (cat, ch), df in series_dict.items():
        s = df[TARGET_COL].dropna()
        rows.append({
            "category":  cat,
            "channel":   ch,
            "n_obs":     int(len(s)),
            "mean":      round(float(s.mean()), 0),
            "std":       round(float(s.std()), 0),
            "cv":        round(float(s.std() / s.mean()), 3) if s.mean() != 0 else None,
            "min":       round(float(s.min()), 0),
            "max":       round(float(s.max()), 0),
            "date_from": str(s.index.min().date()),
            "date_to":   str(s.index.max().date()),
        })
    return rows


def get_series_history(series_dict: dict, category: str, channel: str) -> dict:
    key = (category, channel)
    if key not in series_dict:
        return {}
    s = series_dict[key][TARGET_COL].dropna()
    return {
        "category": category,
        "channel":  channel,
        "dates":    [str(d.date()) for d in s.index],
        "values":   [round(float(v), 2) for v in s.values],
    }


def get_stl_decomposition(series_dict: dict, category: str, channel: str) -> dict:

    key = (category, channel)
    if key not in series_dict:
        return {}

    s = series_dict[key][TARGET_COL].dropna()
    if len(s) < 2 * SEASONAL_PERIOD:
        return {"error": "Недостаточно точек для STL-декомпозиции"}

    res = STL(s, period=SEASONAL_PERIOD, robust=True).fit()

    var_r  = float(res.resid.var())
    var_sr = float((res.seasonal + res.resid).var())
    var_tr = float((res.trend    + res.resid).var())

    Fs = max(0.0, 1 - var_r / var_sr) if var_sr > 0 else 0.0
    Ft = max(0.0, 1 - var_r / var_tr) if var_tr > 0 else 0.0
    r2 = float(1 - res.resid.pow(2).sum() / (s - s.mean()).pow(2).sum())

    dates = [str(d.date()) for d in s.index]
    return {
        "category":    category,
        "channel":     channel,
        "dates":       dates,
        "observed":    [round(float(v), 2) for v in s.values],
        "trend":       [round(float(v), 2) for v in res.trend.values],
        "seasonal":    [round(float(v), 2) for v in res.seasonal.values],
        "resid":       [round(float(v), 2) for v in res.resid.values],
        "metrics": {
            "Fs_seasonal": round(Fs, 4),
            "Ft_trend":    round(Ft, 4),
            "R2":          round(r2, 4),
            "seas_amplitude": round(float(res.seasonal.max() - res.seasonal.min()), 0),
            "trend_slope_mo": round(
                float((res.trend.iloc[-1] - res.trend.iloc[0]) / len(res.trend)), 0
            ),
        },
    }



def get_stationarity(series_dict: dict, category: str, channel: str) -> dict:

    key = (category, channel)
    if key not in series_dict:
        return {}

    s = series_dict[key][TARGET_COL].dropna()

    variants = {
        "original": s,
        "d=1":      s.diff().dropna(),
        "D=1":      s.diff(SEASONAL_PERIOD).dropna() if len(s) > SEASONAL_PERIOD else None,
        "d=1+D=1":  s.diff().diff(SEASONAL_PERIOD).dropna() if len(s) > SEASONAL_PERIOD + 1 else None,
    }

    results = {}
    for label, s_var in variants.items():
        if s_var is None or len(s_var) < 12:
            continue
        results[label] = _run_adf_kpss(s_var)

    return {"category": category, "channel": channel, "tests": results}


def _run_adf_kpss(s: pd.Series) -> dict:
    adf_stat, adf_p, _, _, adf_crit, _ = adfuller(s, autolag="AIC")
    kpss_stat, kpss_p, _, _ = kpss(s, regression="c", nlags="auto")

    stat_adf  = adf_p  < 0.05
    stat_kpss = kpss_p >= 0.05

    if stat_adf and stat_kpss:
        conclusion = "СТАЦИОНАРЕН"
    elif not stat_adf and not stat_kpss:
        conclusion = "НЕСТАЦИОНАРЕН"
    elif stat_adf and not stat_kpss:
        conclusion = "ТРЕНД-СТАЦИОНАРЕН"
    else:
        conclusion = "НЕОПРЕДЕЛЁННО"

    return {
        "adf_stat":     round(float(adf_stat), 4),
        "adf_p":        round(float(adf_p), 4),
        "adf_1pct":     round(float(adf_crit["1%"]), 4),
        "kpss_stat":    round(float(kpss_stat), 4),
        "kpss_p":       round(float(kpss_p), 4),
        "conclusion":   conclusion,
        "is_stationary": bool(stat_adf and stat_kpss),
    }



def get_acf_pacf(series_dict: dict, category: str, channel: str,
                  n_lags: int = 36, diff: str = "original") -> dict:

    key = (category, channel)
    if key not in series_dict:
        return {}

    s = series_dict[key][TARGET_COL].dropna()

    if diff == "d=1":
        s = s.diff().dropna()
    elif diff == "D=1":
        s = s.diff(SEASONAL_PERIOD).dropna()
    elif diff == "d=1+D=1":
        s = s.diff().diff(SEASONAL_PERIOD).dropna()

    if len(s) < 20:
        return {"error": "Недостаточно точек"}

    nl = min(n_lags, len(s) // 2 - 1)
    cb = float(1.96 / np.sqrt(len(s)))

    acf_vals  = acf(s,  nlags=nl, fft=True).tolist()
    pacf_vals = pacf(s, nlags=nl).tolist()

    return {
        "category":   category,
        "channel":    channel,
        "diff":       diff,
        "lags":       list(range(len(acf_vals))),
        "acf":        [round(v, 4) for v in acf_vals],
        "pacf":       [round(v, 4) for v in pacf_vals],
        "conf_bound": round(cb, 4),
        "seasonal_lags": [SEASONAL_PERIOD * i for i in range(1, nl // SEASONAL_PERIOD + 1)],
    }


def analyze_ccf(series_dict: dict, max_lag: int = 12,
                n_sample: int = 8) -> tuple:
    """
    Кросс-корреляционный анализ экзогенных переменных с таргетом.
    Возвращает: (ccf_df, ccf_summary) — детали и сводка по переменным.
    """
    import random
    random.seed(RANDOM_STATE)
    keys = random.sample(list(series_dict.keys()), min(n_sample, len(series_dict)))

    rows = []
    for key in keys:
        df_ = series_dict[key]
        y = df_[TARGET_COL].dropna()
        y_norm = (y - y.mean()) / (y.std() + 1e-8)
        for col in EXOG_COLS:
            if col not in df_.columns:
                continue
            x = df_[col].dropna()
            common = y.index.intersection(x.index)
            if len(common) < 20:
                continue
            x_norm = (x[common] - x[common].mean()) / (x[common].std() + 1e-8)
            y_norm_c = y_norm[common]
            ccf_vals = ccf(x_norm.values, y_norm_c.values, nlags=max_lag, alpha=None)
            best_lag = int(np.argmax(np.abs(ccf_vals[1:]))) + 1
            best_corr = float(ccf_vals[best_lag])
            rows.append({
                "category": key[0], "channel": key[1], "exog_col": col,
                "best_lag": best_lag, "best_ccf": round(best_corr, 4),
                "ccf_vals": ccf_vals[1:max_lag + 1].tolist(),
            })

    ccf_df = pd.DataFrame(rows)
    summary = ccf_df.groupby("exog_col").agg(
        median_best_lag=("best_lag", "median"),
        mean_best_ccf=("best_ccf", lambda x: np.abs(x).mean()),
        pct_significant=("best_ccf", lambda x: (np.abs(x) > 0.3).mean()),
    ).round(3).sort_values("mean_best_ccf", ascending=False)

    print("\n=== CCF: медианный лаг и сила связи ===")
    print(summary.to_string())
    return ccf_df, summary


def build_ccf_lag_ranges(ccf_summary: pd.DataFrame) -> dict:
    """
    Строит словарь {col: [lag1, lag2, ...]} на основе CCF анализа.
    """
    lag_ranges = {}
    for col in EXOG_COLS:
        if col in ccf_summary.index:
            med_lag = int(ccf_summary.loc[col, "median_best_lag"])
            lag_ranges[col] = list(range(1, min(med_lag + 2, 7)))
        else:
            lag_ranges[col] = [1, 2]
    return lag_ranges


def plot_ccf_heatmap(ccf_df: pd.DataFrame, max_lag: int = 8,
                     save_path: str = None):
    rows = []
    for _, r in ccf_df.iterrows():
        for lag, val in enumerate(r["ccf_vals"][:max_lag], 1):
            rows.append({"exog_col": r["exog_col"], "lag": lag, "ccf": abs(val)})
    pivot = pd.DataFrame(rows).groupby(["exog_col", "lag"])["ccf"].mean().unstack()
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="YlOrRd",
                vmin=0, vmax=1, linewidths=0.5, ax=ax)
    ax.set_title("Средняя |CCF| по лагам (экзогенные vs MT_Volume KG)")
    ax.set_xlabel("Лаг (месяцев)")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=130, bbox_inches="tight")


def build_features_single(df_: pd.DataFrame, key: tuple,
                          cat_encoder: dict, ch_encoder: dict,
                          ccf_lag_ranges: dict) -> pd.DataFrame:
    """
    Строит матрицу признаков для одного ряда.

    Таргет хранится в лог-масштабе log1p(y).
    Все признаки сдвинуты shift(1) минимум — нет leakage.
    y_normalized: Z-score по скользящему окну из shift(1) — без leakage.
    """
    s = df_[TARGET_COL].dropna().copy()
    s_log = np.log1p(s)
    feat = pd.DataFrame(index=s.index)

    # Лаги логарифмированной целевой
    for lag in [1, 2, 3, 6, 12, 13, 24]:
        feat[f"y_lag{lag}"] = s_log.shift(lag)

    # Rolling по логарифму (shift(1) — нет leakage)
    s_log_sh = s_log.shift(1)
    for w in [3, 6, 12]:
        feat[f"rolling_mean_{w}"] = s_log_sh.rolling(w).mean()
        feat[f"rolling_std_{w}"] = s_log_sh.rolling(w).std()

    feat["rolling_min_6"] = s_log_sh.rolling(6).min()
    feat["rolling_max_6"] = s_log_sh.rolling(6).max()
    feat["rolling_pos_6"] = (
            (s_log_sh - feat["rolling_min_6"]) /
            (feat["rolling_max_6"] - feat["rolling_min_6"] + 1e-8)
    )
    feat["ewm_03"] = s_log_sh.ewm(alpha=0.3, adjust=False).mean()
    feat["ewm_07"] = s_log_sh.ewm(alpha=0.7, adjust=False).mean()

    # Per-series нормализация — используем s_log_sh (без leakage)
    s_mean = s_log_sh.rolling(12, min_periods=6).mean()
    s_std = s_log_sh.rolling(12, min_periods=6).std().clip(lower=1e-8)
    feat["y_normalized"] = (s_log_sh - s_mean) / s_std

    # YoY и MoM в лог-пространстве
    feat["yoy_diff"] = s_log.shift(1) - s_log.shift(13)
    feat["mom_diff"] = s_log.shift(1) - s_log.shift(2)

    # Лаги экзогенных (лаги из CCF)
    for col in EXOG_COLS:
        if col not in df_.columns:
            continue
        x = df_[col]
        col_clean = col.replace(" ", "_").replace("/", "_")
        if col in ["NT_Universe", "NT_CWD"]:
            x = np.log1p(x)
            col_clean = f"log_{col_clean}"
        for lag in ccf_lag_ranges.get(col, [1, 2]):
            feat[f"{col_clean}_lag{lag}"] = x.shift(lag)

    # Тренд
    feat["t"] = np.arange(len(s))
    feat["t_squared"] = feat["t"] ** 2
    feat["trend_approx"] = s_log_sh.rolling(12, min_periods=6).mean()

    # Календарь
    feat["month"] = s.index.month
    feat["month_sin"] = np.sin(2 * np.pi * s.index.month / 12)
    feat["month_cos"] = np.cos(2 * np.pi * s.index.month / 12)
    feat["quarter"] = s.index.quarter
    feat["quarter_sin"] = np.sin(2 * np.pi * s.index.quarter / 4)
    feat["quarter_cos"] = np.cos(2 * np.pi * s.index.quarter / 4)
    feat["year"] = s.index.year - s.index.year.min()
    feat["is_q4"] = (s.index.month >= 10).astype(int)
    feat["is_summer"] = s.index.month.isin([6, 7, 8]).astype(int)

    # COVID дамми
    feat["covid"] = (
            (s.index >= COVID_START) & (s.index <= COVID_END)
    ).astype(int)
    feat["post_covid"] = (
            (s.index > COVID_END) & (s.index <= "2021-12-01")
    ).astype(int)

    # Идентификаторы ряда
    feat["category_enc"] = cat_encoder[key[0]]
    feat["channel_enc"] = ch_encoder[key[1]]
    feat["series_id"] = feat["category_enc"] * 10 + feat["channel_enc"]

    # Целевая в лог-масштабе
    feat[TARGET_COL] = s_log
    return feat


def build_global_dataset(series_dict: dict,
                         ccf_lag_ranges: dict) -> tuple:
    """
    Собирает глобальный датасет из всех рядов.
    Возвращает: (global_df, feature_cols, cat_enc, ch_enc)
    """
    categories = sorted({k[0] for k in series_dict})
    channels = sorted({k[1] for k in series_dict})
    cat_enc = {c: i for i, c in enumerate(categories)}
    ch_enc = {c: i for i, c in enumerate(channels)}

    all_frames = []
    for key, df_ in series_dict.items():
        feat = build_features_single(df_, key, cat_enc, ch_enc, ccf_lag_ranges)
        feat["_category"] = key[0]
        feat["_channel"] = key[1]
        feat["_date"] = feat.index
        all_frames.append(feat)

    global_df = pd.concat(all_frames, axis=0).reset_index(drop=True)
    feature_cols = [c for c in global_df.columns
                    if c not in [TARGET_COL, "_category", "_channel", "_date"]]

    # Убираем строки где слишком много NaN в лаговых признаках
    lag_cols = [c for c in feature_cols if "lag" in c or "rolling" in c]
    thresh = int(len(lag_cols) * 0.7)
    global_df = global_df.dropna(
        subset=lag_cols, thresh=thresh
    ).reset_index(drop=True)

    print(f"\nГлобальный датасет: {global_df.shape[0]} строк, "
          f"{len(feature_cols)} признаков")
    print("Целевая переменная: log1p(MT_Volume KG)")
    return global_df, feature_cols, cat_enc, ch_enc


# =============================================================================
# WALK-FORWARD CV РАЗБИВКА
# =============================================================================

def make_cv_folds(global_df: pd.DataFrame,
                  n_folds: int = N_FOLDS,
                  test_size: int = TEST_SIZE) -> list:
    """
    Создаёт walk-forward CV фолды.
    min_train = 2 * SEASONAL_PERIOD (не добавляем HORIZON — иначе только 2 фолда).

    Возвращает список: [(train_mask, test_mask, fold_num), ...]
    """
    dates = sorted(global_df["_date"].unique())
    n = len(dates)
    min_train = 2 * SEASONAL_PERIOD
    folds = []

    for fold in range(n_folds):
        test_end_idx = n - fold * test_size
        test_start_idx = test_end_idx - test_size
        if test_start_idx < min_train:
            break
        train_dates = dates[:test_start_idx]
        test_dates = dates[test_start_idx:test_end_idx]
        folds.append((
            global_df["_date"].isin(train_dates),
            global_df["_date"].isin(test_dates),
            fold + 1,
        ))

    folds.reverse()
    print(f"\nСоздано {len(folds)} фолдов:")
    for tr, te, num in folds:
        tr_dates = sorted(global_df.loc[tr, "_date"].unique())
        te_dates = sorted(global_df.loc[te, "_date"].unique())
        print(f"  Фолд {num}: train {tr_dates[0].date()}—{tr_dates[-1].date()} "
              f"({len(tr_dates)} мес.) | "
              f"test {te_dates[0].date()}—{te_dates[-1].date()}")
    return folds