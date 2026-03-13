import random
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import acf, pacf

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