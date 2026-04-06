import json
import pickle
import random
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import grangercausalitytests
from pmdarima import auto_arima
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import LinearRegression

from src.config.settings import (
    TARGET_COL, EXOG_CANDIDATES, SEASONAL_PERIOD, RANDOM_STATE,
    SARIMA_DIR, RESULTS_DIR,
)
from src.utils.metrics import compute_metrics

random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

MANIFEST_PATH = SARIMA_DIR / "manifest.json"


def _get_exog_cols(df: pd.DataFrame) -> list:
    return [c for c in EXOG_CANDIDATES if c in df.columns and df[c].notna().sum() > 10]


def _compute_vif(df_exog: pd.DataFrame) -> pd.Series:
    cols = df_exog.columns.tolist()
    X    = df_exog.values
    vifs = {}
    for i, col in enumerate(cols):
        X_rest = np.delete(X, i, axis=1)
        if X_rest.shape[1] == 0:
            vifs[col] = 1.0
            continue
        try:
            r2 = LinearRegression().fit(X_rest, X[:, i]).score(X_rest, X[:, i])
            vifs[col] = round(1 / (1 - r2) if r2 < 1 else np.inf, 2)
        except Exception:
            vifs[col] = np.nan
    return pd.Series(vifs, name="VIF")


def _compute_max_exog(n_obs: int, d: int = 1, D: int = 1, arma_params: int = 5) -> int:
    n_eff = n_obs - d - D * SEASONAL_PERIOD
    return max(0, min(int(np.floor(n_eff * 0.20)) - arma_params, 3))


def _forecast_exog_via_sarima(exog_series: pd.Series, horizon: int) -> pd.Series:
    future_idx = pd.date_range(
        exog_series.index[-1] + pd.DateOffset(months=1), periods=horizon, freq="MS"
    )
    try:
        m  = auto_arima(exog_series.dropna(), m=SEASONAL_PERIOD, seasonal=True,
                        max_p=2, max_q=2, max_P=1, max_Q=1, d=None, D=None,
                        information_criterion="aic", stepwise=True,
                        error_action="ignore", suppress_warnings=True)
        return pd.Series(m.predict(n_periods=horizon), index=future_idx, name=exog_series.name)
    except Exception:
        val = exog_series.tail(12).mean()
        return pd.Series(val, index=future_idx, name=exog_series.name)


def _analyze_exog_correlation(series_dict: dict) -> pd.DataFrame:
    rows = []
    for (cat, ch), df in series_dict.items():
        y = df[TARGET_COL].dropna()
        for col in _get_exog_cols(df):
            x = df[col].dropna()
            common = y.index.intersection(x.index)
            if len(common) < 10:
                continue
            r  = np.corrcoef(y[common], x[common])[0, 1]
            mi = mutual_info_regression(
                x[common].values.reshape(-1, 1), y[common].values,
                random_state=RANDOM_STATE
            )[0]
            rows.append({"category": cat, "channel": ch,
                         "exog_col": col, "pearson_r": round(r, 4),
                         "mutual_info": round(mi, 4)})
    return pd.DataFrame(rows)


def _granger_test(series_dict: dict, max_lag: int = 4) -> pd.DataFrame:
    rows = []
    for (cat, ch), df in series_dict.items():
        y = df[TARGET_COL].dropna()
        for col in _get_exog_cols(df):
            x = df[col].dropna()
            common = y.index.intersection(x.index)
            if len(common) < 20:
                continue
            data = pd.DataFrame({"y": y[common], "x": x[common]}).dropna()
            try:
                gc    = grangercausalitytests(data[["y", "x"]], maxlag=max_lag, verbose=False)
                min_p = min(gc[lag][0]["ssr_ftest"][1] for lag in range(1, max_lag + 1))
                rows.append({"category": cat, "channel": ch, "exog_col": col,
                             "granger_min_p": round(min_p, 4),
                             "granger_significant": min_p < 0.05})
            except Exception:
                pass
    return pd.DataFrame(rows)


def _rf_importance(series_dict: dict, n_lags: int = 3) -> pd.DataFrame:
    rows = []
    for (cat, ch), df in series_dict.items():
        exog_cols = _get_exog_cols(df)
        if not exog_cols:
            continue
        work = df[[TARGET_COL] + exog_cols].dropna()
        if len(work) < 20:
            continue

        feat = pd.DataFrame(index=work.index)
        for lag in range(1, n_lags + 1):
            feat[f"y_lag{lag}"] = work[TARGET_COL].shift(lag)
        for col in exog_cols:
            feat[col] = work[col]
            for lag in range(1, n_lags + 1):
                feat[f"{col}_lag{lag}"] = work[col].shift(lag)
        feat["y"] = work[TARGET_COL]
        feat = feat.dropna()
        if len(feat) < 10:
            continue

        X, y_rf = feat.drop(columns=["y"]), feat["y"]
        rf = RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE)
        rf.fit(X, y_rf)
        imp = pd.Series(rf.feature_importances_, index=X.columns)

        for col in exog_cols:
            total = imp[[c for c in imp.index if c.startswith(col)]].sum()
            rows.append({"category": cat, "channel": ch,
                         "exog_col": col, "rf_importance": round(float(total), 4)})
    return pd.DataFrame(rows)


def select_exog_variables(series_dict: dict,
                           pearson_thresh: float = 0.3,
                           granger_p_thresh: float = 0.05,
                           rf_thresh: float = 0.05,
                           min_score: int = 2,
                           vif_thresh: float = 10.0) -> dict:
    corr_df = _analyze_exog_correlation(series_dict)
    gc_df   = _granger_test(series_dict)
    rf_df   = _rf_importance(series_dict)

    all_pairs = set()
    for df_ in [corr_df, gc_df, rf_df]:
        if not df_.empty:
            for _, r in df_.iterrows():
                all_pairs.add((r["category"], r["channel"]))

    all_vars = set(corr_df["exog_col"].unique()) if not corr_df.empty else set()
    per_series_exog = {}

    for pair in sorted(all_pairs):
        cat, ch = pair
        n_obs   = len(series_dict[pair][TARGET_COL].dropna()) if pair in series_dict else 65
        k_max   = _compute_max_exog(n_obs)

        scored = []
        for var in sorted(all_vars):
            score, pearson_val = 0, 0.0

            if not corr_df.empty:
                m = corr_df[(corr_df["category"] == cat) & (corr_df["channel"] == ch)
                            & (corr_df["exog_col"] == var)]["pearson_r"].abs()
                if len(m):
                    pearson_val = m.values[0]
                    if pearson_val >= pearson_thresh:
                        score += 1

            if not gc_df.empty:
                m = gc_df[(gc_df["category"] == cat) & (gc_df["channel"] == ch)
                          & (gc_df["exog_col"] == var)]["granger_min_p"]
                if len(m) and m.values[0] < granger_p_thresh:
                    score += 1

            if not rf_df.empty:
                m = rf_df[(rf_df["category"] == cat) & (rf_df["channel"] == ch)
                          & (rf_df["exog_col"] == var)]["rf_importance"]
                if len(m) and m.values[0] >= rf_thresh:
                    score += 1

            if score >= min_score:
                scored.append((var, score, pearson_val))

        scored.sort(key=lambda x: (x[1], x[2]), reverse=True)
        after_stage2 = [v for v, _, _ in scored[:k_max]]

        after_stage3 = list(after_stage2)
        if len(after_stage3) >= 2 and pair in series_dict:
            df_s  = series_dict[pair]
            y_idx = df_s[TARGET_COL].dropna().index
            avail = [c for c in after_stage3
                     if c in df_s.columns and df_s.loc[y_idx, c].notna().all()]
            if len(avail) >= 2:
                exog_data = df_s.loc[y_idx, avail]
                while len(avail) >= 2:
                    vif_s = _compute_vif(exog_data[avail])
                    if vif_s.max() <= vif_thresh:
                        break
                    avail.remove(vif_s.idxmax())
                after_stage3 = avail

        per_series_exog[pair] = after_stage3

    n_sarimax = sum(1 for v in per_series_exog.values() if v)
    print(f"exog: SARIMA: {len(per_series_exog) - n_sarimax}, SARIMAX: {n_sarimax}")
    return per_series_exog


def find_global_order(series_dict: dict, per_series_exog: dict, n_sample: int = 5) -> dict:
    keys = random.sample(list(series_dict.keys()), min(n_sample, len(series_dict)))
    p_list, d_list, q_list, P_list, D_list, Q_list = [], [], [], [], [], []

    for key in keys:
        df_ = series_dict[key]
        y   = df_[TARGET_COL].dropna()
        exog = None
        key_exog = per_series_exog.get(key, [])
        if key_exog:
            avail = [c for c in key_exog if c in df_.columns and df_[c].notna().all()]
            if avail:
                exog = df_.loc[y.index, avail]
        try:
            model = auto_arima(
                y, exogenous=exog, m=SEASONAL_PERIOD, seasonal=True,
                d=None, D=None, max_p=3, max_q=3, max_P=2, max_Q=2,
                information_criterion="aic", stepwise=True,
                error_action="ignore", suppress_warnings=True,
                random_state=RANDOM_STATE,
            )
            o, s = model.order, model.seasonal_order
            p_list.append(o[0]); d_list.append(o[1]); q_list.append(o[2])
            P_list.append(s[0]); D_list.append(s[1]); Q_list.append(s[2])
        except Exception as e:
            print(f"warn:  {key}: {e}")

    def mode(lst):
        return max(set(lst), key=lst.count) if lst else 0

    order = {k: mode(v) for k, v in zip(
        ["p", "d", "q", "P", "D", "Q"],
        [p_list, d_list, q_list, P_list, D_list, Q_list]
    )}
    print(f"auto_arima: Глобальный порядок: "
          f"SARIMA({order['p']},{order['d']},{order['q']})"
          f"({order['P']},{order['D']},{order['Q']})[12]")
    return order


def train_all_models(series_dict: dict, per_series_exog: dict, global_order: dict,
                      use_individual_order: bool = True, test_size: int = 12) -> dict:
    results = {}
    total   = len(series_dict)

    for i, (key, df_) in enumerate(series_dict.items(), 1):
        print(f"\n[{i}/{total}] {key[0]} | {key[1]}")
        y = df_[TARGET_COL].dropna()

        if len(y) < 2 * SEASONAL_PERIOD + test_size:
            print(f"Пропущен: мало точек ({len(y)})")
            continue

        exog_full, exog_cols_used = None, []
        candidates = per_series_exog.get(key, [])
        if candidates:
            avail = [c for c in candidates
                     if c in df_.columns and df_.loc[y.index, c].notna().all()]
            if avail:
                exog_full      = df_.loc[y.index, avail]
                exog_cols_used = avail

        y_train    = y.iloc[:-test_size]
        y_test     = y.iloc[-test_size:]
        exog_train = exog_full.iloc[:-test_size] if exog_full is not None else None
        exog_test  = exog_full.iloc[-test_size:] if exog_full is not None else None

        if use_individual_order:
            try:
                am = auto_arima(
                    y_train, exogenous=exog_train,
                    m=SEASONAL_PERIOD, seasonal=True, d=None, D=None,
                    max_p=3, max_q=3, max_P=2, max_Q=2,
                    information_criterion="aic", stepwise=True,
                    error_action="ignore", suppress_warnings=True,
                    random_state=RANDOM_STATE,
                )
                order, seasonal_order = am.order, am.seasonal_order
            except Exception as e:
                print(f"Warn: auto_arima: {e}. Используем глобальный порядок.")
                order = (global_order["p"], global_order["d"], global_order["q"])
                seasonal_order = (global_order["P"], global_order["D"],
                                  global_order["Q"], SEASONAL_PERIOD)
        else:
            order = (global_order["p"], global_order["d"], global_order["q"])
            seasonal_order = (global_order["P"], global_order["D"],
                              global_order["Q"], SEASONAL_PERIOD)

        try:
            fitted = SARIMAX(
                y_train, exog=exog_train, order=order, seasonal_order=seasonal_order,
                enforce_stationarity=False, enforce_invertibility=False,
            ).fit(disp=False, maxiter=300)
        except Exception as e:
            print(f"Error: {e}")
            continue

        forecast_test = fitted.forecast(steps=test_size, exog=exog_test)
        metrics = compute_metrics(y_test.values, forecast_test.values)

        lb = acorr_ljungbox(fitted.resid, lags=[12], return_df=True)
        lb_p = lb["lb_pvalue"].values[0]
        metrics["lb_p"]     = round(float(lb_p), 4)
        metrics["resid_ok"] = bool(lb_p > 0.05)

        model_type = f"SARIMAX ({len(exog_cols_used)} экзог.)" if exog_cols_used else "SARIMA"
        print(f"  {model_type}  MAPE={metrics['mape']:.1f}%  "
              f"Ljung-Box {'OK' if metrics['resid_ok'] else '!NOT OK!'}")

        results[key] = {
            "model": fitted, "order": order, "seasonal_order": seasonal_order,
            "exog_cols": exog_cols_used, "model_type": model_type,
            "y_train": y_train, "y_test": y_test,
            "forecast_test": forecast_test, "metrics": metrics, "aic": fitted.aic,
        }

    return results


def train_seasonal_naive(series_dict: dict, test_size: int = 12) -> dict:
    results = {}
    for key, df_ in series_dict.items():
        y = df_[TARGET_COL].dropna()
        if len(y) < SEASONAL_PERIOD + test_size:
            continue

        y_train, y_test = y.iloc[:-test_size], y.iloc[-test_size:]
        fc = np.array([
            y_train.iloc[-(SEASONAL_PERIOD - h % SEASONAL_PERIOD)]
            if len(y_train) >= SEASONAL_PERIOD else y_train.iloc[-1]
            for h in range(test_size)
        ])
        forecast_test = pd.Series(fc, index=y_test.index)
        metrics = compute_metrics(y_test.values, forecast_test.values)
        metrics["lb_p"] = np.nan
        metrics["resid_ok"] = False

        results[key] = {
            "model": None, "order": (0, 0, 0),
            "seasonal_order": (0, 0, 0, SEASONAL_PERIOD),
            "exog_cols": [], "model_type": "Naive",
            "y_train": y_train, "y_test": y_test,
            "forecast_test": forecast_test, "metrics": metrics, "aic": np.nan,
        }
    return results


def train_sarima_no_exog(series_dict: dict, global_order: dict,
                          test_size: int = 12) -> dict:
    empty_exog = {key: [] for key in series_dict}
    return train_all_models(series_dict, empty_exog, global_order,
                            use_individual_order=False, test_size=test_size)



def select_best_and_save(results_sarimax: dict, results_sarima: dict,
                          results_naive: dict, metric: str = "mape") -> dict:
    all_keys = set(results_sarimax) | set(results_sarima) | set(results_naive)
    manifest = {}

    for key in sorted(all_keys):
        candidates = {
            "Naive":  results_naive.get(key),
            "SARIMA":  results_sarima.get(key),
            "SARIMAX": results_sarimax.get(key)}
        candidates = {k: v for k, v in candidates.items() if v is not None}

        best_label = min(
            candidates,
            key=lambda k: candidates[k]["metrics"].get(metric, np.inf) or np.inf)
        best_res = candidates[best_label]

        safe_name  = f"{key[0]}_{key[1]}".replace(" ", "_").replace("/", "-")
        model_file = str(SARIMA_DIR / f"{safe_name}.pkl")

        if best_res["model"] is not None:
            with open(model_file, "wb") as f:
                pickle.dump({
                    "fitted_model":   best_res["model"],
                    "order":          best_res["order"],
                    "seasonal_order": best_res["seasonal_order"],
                    "exog_cols":      best_res["exog_cols"],
                }, f)

        manifest_key = f"{key[0]}|{key[1]}"
        manifest[manifest_key] = {
            "category":      key[0],
            "channel":       key[1],
            "best_model":    best_label,
            "order":         list(best_res["order"]),
            "seasonal_order": list(best_res["seasonal_order"]),
            "exog_cols":     best_res["exog_cols"],
            "metrics": {
                k: round(float(v), 4) if isinstance(v, (int, float, np.floating))
                   and not np.isnan(v) else None
                for k, v in best_res["metrics"].items()
                if k not in ("lb_p", "resid_ok")
            },
            "mape":       round(float(best_res["metrics"].get("mape") or np.inf), 2),
            "model_file": model_file if best_res["model"] is not None else None,
        }

        print(f"  {key[0][:20]:<20} | {key[1]:<24} → {best_label:<8}  "
              f"MAPE={manifest[manifest_key]['mape']:.1f}%")

    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"\n manifest сохранён в {MANIFEST_PATH}")
    return manifest


def train_arima(series_dict: dict, test_size: int = 12) -> dict:
    """
    Обучает чистую ARIMA(p,d,q) для каждого ряда — без сезонной компоненты,
    без экзогенных переменных.

    Отличие от SARIMA:
      SARIMA = SARIMAX(p,d,q)(P,D,Q)[12] — учитывает сезонность
      ARIMA  = ARIMA(p,d,q)              — только несезонная структура

    Используется для сравнения в консольном отчёте (не логируется в MLflow).
    Порядок подбирается индивидуально через auto_arima с seasonal=False.
    """
    results = {}
    total   = len(series_dict)
    print(f"\nARIMA: обучение {total} рядов...")

    for i, (key, df_) in enumerate(series_dict.items(), 1):
        cat, ch = key
        y = df_[TARGET_COL].dropna()

        if len(y) < 2 * test_size:
            print(f"  [{i}/{total}] {cat} | {ch} — пропущен (мало данных: {len(y)})")
            continue

        y_train = y.iloc[:-test_size]
        y_test  = y.iloc[-test_size:]

        try:
            am = auto_arima(
                y_train,
                seasonal=False,          # ARIMA, не SARIMA
                d=None,                  # автоматический порядок интегрирования
                max_p=3, max_q=3,
                information_criterion="aic",
                stepwise=True,
                error_action="ignore",
                suppress_warnings=True,
                random_state=RANDOM_STATE,
            )
            order = am.order  # (p, d, q)
        except Exception as e:
            print(f"  [{i}/{total}] {cat} | {ch} — auto_arima ошибка: {e}, fallback ARIMA(1,1,1)")
            order = (1, 1, 1)

        try:
            fitted = SARIMAX(
                y_train,
                order=order,
                seasonal_order=(0, 0, 0, 0),  # без сезонной компоненты
                enforce_stationarity=False,
                enforce_invertibility=False,
            ).fit(disp=False, maxiter=300)
        except Exception as e:
            print(f"  [{i}/{total}] {cat} | {ch} — SARIMAX fit ошибка: {e}")
            continue

        forecast_test = fitted.forecast(steps=test_size)
        metrics       = compute_metrics(y_test.values, forecast_test.values)

        print(f"  [{i}/{total}] {cat[:20]:<20} | {ch:<24}  "
              f"ARIMA{order}  MAPE={metrics['mape']:.1f}%")

        results[key] = {
            "order":          order,
            "seasonal_order": (0, 0, 0, 0),
            "model_type":     "ARIMA",
            "y_train":        y_train,
            "y_test":         y_test,
            "forecast_test":  forecast_test,
            "metrics":        metrics,
            "aic":            fitted.aic,
        }

    return results


def compare_all_models(
    results_sarimax: dict,
    results_sarima:  dict,
    results_naive:   dict,
    results_arima:   dict,
    metric: str = "mape",
) -> pd.DataFrame:
    """
    Строит сводную таблицу MAPE для всех четырёх моделей по каждому ряду.
    Выводит в консоль и возвращает DataFrame.
    Не логирует в MLflow.
    """
    all_keys = (
        set(results_sarimax)
        | set(results_sarima)
        | set(results_naive)
        | set(results_arima)
    )

    rows = []
    for key in sorted(all_keys):
        cat, ch = key
        row = {"Категория": cat, "Канал": ch}

        for label, res_dict in [
            ("Naive",   results_naive),
            ("ARIMA",   results_arima),
            ("SARIMA",  results_sarima),
            ("SARIMAX", results_sarimax),
        ]:
            res = res_dict.get(key)
            val = None
            if res is not None:
                val = res["metrics"].get(metric)
                if isinstance(val, float) and np.isnan(val):
                    val = None
            row[f"{label} MAPE,%"] = round(float(val), 2) if val is not None else None

        # Победитель по метрике среди доступных моделей
        candidates = {
            label: row[f"{label} MAPE,%"]
            for label in ("Naive", "ARIMA", "SARIMA", "SARIMAX")
            if row.get(f"{label} MAPE,%") is not None
        }
        row["Лучшая"] = min(candidates, key=candidates.get) if candidates else "—"
        rows.append(row)

    df = pd.DataFrame(rows)

    # ── Вывод в консоль ──────────────────────────────────────────────────────
    sep = "=" * 100
    print(f"\n{sep}")
    print("СРАВНЕНИЕ ВСЕХ МОДЕЛЕЙ (MAPE на тестовом периоде, %)")
    print(sep)
    print(df.to_string(index=False))

    # Сводка по победителям
    print(f"\n{'─' * 50}")
    print("Победитель по рядам:")
    for label in ("Naive", "ARIMA", "SARIMA", "SARIMAX"):
        n = (df["Лучшая"] == label).sum()
        if n:
            print(f"  {label:<10}: {n} рядов")

    # Медианы
    print(f"\nМедианный MAPE по всем рядам:")
    for label in ("Naive", "ARIMA", "SARIMA", "SARIMAX"):
        col = f"{label} MAPE,%"
        if col in df.columns:
            med = df[col].dropna().median()
            print(f"  {label:<10}: {med:.2f}%")
    print(sep)

    return df


def load_manifest() -> dict:
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Манифест не найден: {MANIFEST_PATH}. Запусти обучение через /train."
        )
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        return json.load(f)



def forecast_from_manifest(series_dict: dict, category: str, channel: str,
                             horizon: int = 12,
                             exog_df: pd.DataFrame = None) -> dict:

    manifest = load_manifest()
    key_str  = f"{category}|{channel}"

    if key_str not in manifest:
        raise ValueError(f"Ряд '{key_str}' не найден в манифесте.")

    info       = manifest[key_str]
    best_model = info["best_model"]
    key        = (category, channel)

    if key not in series_dict:
        raise ValueError(f"Ряд '{key_str}' не найден в данных.")

    df_   = series_dict[key]
    y     = df_[TARGET_COL].dropna()
    future_idx = pd.date_range(
        y.index[-1] + pd.DateOffset(months=1), periods=horizon, freq="MS")

    if best_model == "Naive":
        fc = np.array([y.iloc[-(SEASONAL_PERIOD - h % SEASONAL_PERIOD)]
                       for h in range(horizon)])
        sigma = np.std(y.values[SEASONAL_PERIOD:] - y.values[:-SEASONAL_PERIOD])
        return _build_response(
            category, channel, "Naive",
            f"Naive[{SEASONAL_PERIOD}]",
            future_idx, fc,
            lower=fc-1.28*sigma,
            upper=fc+1.28*sigma,
            info=info)

    # ── SARIMA / SARIMAX ──────────────────────────────────────────────────────
    model_file = info.get("model_file")
    if not model_file or not Path(model_file).exists():
        raise FileNotFoundError(f"Файл модели не найден: {model_file}")

    with open(model_file, "rb") as f:
        saved = pickle.load(f)

    fitted = saved["fitted_model"]
    order  = tuple(saved["order"])
    seasonal_order = tuple(saved["seasonal_order"])
    exog_cols = saved["exog_cols"]

    exog_hist = None
    exog_future = None

    if exog_cols:
        exog_hist = df_[exog_cols]

        if exog_df is not None:
            available = [c for c in exog_cols if c in exog_df.columns]
            missing   = [c for c in exog_cols if c not in exog_df.columns]
            if missing:
                print(f"Переменные не найдены в файле: {missing}. Прогнозируем авто.")

            proj = {}
            for col in exog_cols:
                if col in exog_df.columns:
                    proj[col] = exog_df[col].values[:horizon]
                else:
                    proj[col] = _forecast_exog_via_sarima(exog_hist[col].dropna(), horizon).values
            exog_future = pd.DataFrame(proj, index=future_idx)
        else:
            proj = {col: _forecast_exog_via_sarima(exog_hist[col].dropna(), horizon)
                    for col in exog_cols}
            exog_future = pd.DataFrame(proj, index=future_idx)

    full_model = SARIMAX(
        y, exog=exog_hist, order=order, seasonal_order=seasonal_order,
        enforce_stationarity=False, enforce_invertibility=False,
    ).fit(disp=False, maxiter=300)

    fc   = full_model.get_forecast(steps=horizon, exog=exog_future)
    mean = fc.predicted_mean
    ci   = fc.conf_int(alpha=0.2)

    strategy = best_model + (" (авто экзог.)" if exog_cols and exog_df is None
                             else " (план экзог.)" if exog_cols else "")

    return _build_response(
        category, channel, strategy,
        f"SARIMA{order}x{seasonal_order[:3]}[{SEASONAL_PERIOD}]",
        mean.index, mean.values,
        lower=ci.iloc[:, 0].values,
        upper=ci.iloc[:, 1].values,
        info=info)


def _build_response(category, channel, model_type, model_spec,
                     dates, forecast, lower, upper, info) -> dict:
    return {
        "category":   category,
        "channel":    channel,
        "model_type": model_type,
        "model_spec": model_spec,
        "exog_cols":  info.get("exog_cols", []),
        "train_mape": info.get("mape"),
        "forecast": [
            {
                "date":     str(d.date()) if hasattr(d, "date") else str(d),
                "forecast": round(float(f), 2),
                "lower_80": round(float(lo), 2),
                "upper_80": round(float(hi), 2),
            }
            for d, f, lo, hi in zip(dates, forecast, lower, upper)
        ],
    }