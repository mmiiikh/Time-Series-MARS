"""
Расширенный аналитический модуль.

Блоки:
  1. Циклический анализ (HP-filter, FFT)
  2. Анализ тренда (Манн-Кендалл, структурные изломы, CUSUM)
  3. Углублённый анализ сезонности (субграфики, Kruskal-Wallis)
  4. Анализ аномалий и COVID-влияния
  5. Корреляции между рядами

Все функции возвращают словари/списки — готовы к сериализации в JSON для API.
Графики строятся в Streamlit из этих данных через Plotly.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import signal, stats
from scipy.fft import fft, fftfreq
from statsmodels.tsa.filters.hp_filter import hpfilter
from statsmodels.tsa.stattools import ccf

from src.config.settings import TARGET_COL, SEASONAL_PERIOD, RANDOM_STATE

# Период COVID-шока — используется в нескольких функциях
COVID_START = "2020-03-01"
COVID_END   = "2020-06-01"


# =============================================================================
# 1. ЦИКЛИЧЕСКИЙ АНАЛИЗ
# =============================================================================

def get_hp_filter(series_dict: dict, category: str, channel: str,
                   lamb: int = 1600) -> dict:
    """
    Фильтр Ходрика-Прескотта (HP-filter).

    Разделяет ряд на два компонента:
      - Тренд    — долгосрочное направление движения (сглаженная линия)
      - Цикл     — отклонения от тренда длиннее одного сезона

    Параметр lamb (λ) контролирует гладкость тренда:
      - λ = 1600  → стандарт для квартальных данных
      - λ = 14400 → для месячных данных (рекомендуется)
      - Чем больше λ, тем более гладкий тренд и более выраженный цикл

    КАК ИНТЕРПРЕТИРОВАТЬ:
      - Цикл > 0 → продажи выше долгосрочного тренда (фаза подъёма)
      - Цикл < 0 → продажи ниже долгосрочного тренда (фаза спада)
      - Амплитуда цикла показывает насколько сильны циклические колебания
      - COVID-провал должен быть виден как резкое отрицательное отклонение в 2020
    """
    key = (category, channel)
    if key not in series_dict:
        return {}

    s = series_dict[key][TARGET_COL].dropna()
    if len(s) < 2 * SEASONAL_PERIOD:
        return {"error": "Недостаточно точек для HP-фильтра"}

    # Для месячных данных рекомендуется λ=14400
    lamb_monthly = 14400 if lamb == 1600 else lamb
    cycle, trend = hpfilter(s, lamb=lamb_monthly)

    # Метрики цикла
    cycle_std    = float(cycle.std())
    cycle_max    = float(cycle.max())
    cycle_min    = float(cycle.min())
    cycle_range  = cycle_max - cycle_min

    # Доля времени в положительной фазе (выше тренда)
    pct_above = float((cycle > 0).mean() * 100)

    # Средняя длина цикла (количество месяцев между пересечениями нуля)
    sign_changes = np.where(np.diff(np.sign(cycle.values)))[0]
    avg_half_cycle = float(np.diff(sign_changes).mean()) if len(sign_changes) > 1 else np.nan
    avg_full_cycle = avg_half_cycle * 2 if not np.isnan(avg_half_cycle) else np.nan

    return {
        "category": category,
        "channel":  channel,
        "dates":    [str(d.date()) for d in s.index],
        "observed": [round(float(v), 2) for v in s.values],
        "trend":    [round(float(v), 2) for v in trend.values],
        "cycle":    [round(float(v), 2) for v in cycle.values],
        "metrics": {
            "cycle_std":        round(cycle_std, 2),
            "cycle_amplitude":  round(cycle_range, 2),
            "cycle_max":        round(cycle_max, 2),
            "cycle_min":        round(cycle_min, 2),
            "pct_above_trend":  round(pct_above, 1),
            "avg_cycle_months": round(avg_full_cycle, 1) if not np.isnan(avg_full_cycle) else None,
            "lambda_used":      lamb_monthly,
        },
        "interpretation": {
            "cycle_strength": (
                "Сильный циклический компонент" if cycle_std / s.std() > 0.3
                else "Умеренный циклический компонент" if cycle_std / s.std() > 0.1
                else "Слабый циклический компонент"
            ),
            "current_phase": (
                "Выше тренда (подъём)" if float(cycle.iloc[-1]) > 0
                else "Ниже тренда (спад)"
            ),
        },
    }


def get_fft_analysis(series_dict: dict, category: str, channel: str) -> dict:
    """
    Спектральный анализ через FFT (быстрое преобразование Фурье).

    Ищет доминирующие частоты в ряде — показывает на каких периодах
    сосредоточена наибольшая энергия колебаний.

    КАК ИНТЕРПРЕТИРОВАТЬ:
      - Пик на периоде 12 мес. → сезонный цикл (ожидаем всегда)
      - Пик на периоде 24-36 мес. → среднесрочный бизнес-цикл
      - Пик на периоде > 48 мес. → долгосрочный тренд-цикл
      - Если пик ТОЛЬКО на 12 → ряд чисто сезонный, без длинных циклов
      - Несколько пиков → ряд имеет несколько накладывающихся цикличностей
    """
    key = (category, channel)
    if key not in series_dict:
        return {}

    s = series_dict[key][TARGET_COL].dropna()
    if len(s) < 24:
        return {"error": "Нужно минимум 24 наблюдения для FFT"}

    # Убираем тренд перед FFT чтобы он не доминировал в спектре
    s_detrended = signal.detrend(s.values)

    n      = len(s_detrended)
    yf     = np.abs(fft(s_detrended))[:n // 2]
    xf     = fftfreq(n, d=1)[:n // 2]  # частоты в циклах/месяц

    # Переводим в периоды (месяцы)
    with np.errstate(divide="ignore"):
        periods = np.where(xf > 0, 1 / xf, np.inf)

    # Берём только периоды от 2 до длины ряда
    mask    = (periods >= 2) & (periods <= n)
    periods = periods[mask]
    power   = yf[mask]

    # Топ-5 пиков по мощности
    top_idx  = np.argsort(power)[::-1][:5]
    top_peaks = [
        {
            "period_months": round(float(periods[i]), 1),
            "power":         round(float(power[i]), 2),
            "relative_power": round(float(power[i] / power.max() * 100), 1),
        }
        for i in top_idx if not np.isinf(periods[i])
    ]

    # Нормализуем спектр для графика
    power_norm = (power / power.max() * 100).tolist()

    return {
        "category":   category,
        "channel":    channel,
        "periods":    [round(float(p), 1) for p in periods],
        "power":      [round(float(v), 2) for v in power_norm],
        "top_peaks":  top_peaks,
        "interpretation": {
            "dominant_period": top_peaks[0]["period_months"] if top_peaks else None,
            "has_seasonal":    any(
                abs(p["period_months"] - 12) < 1.5 for p in top_peaks
            ),
            "has_long_cycle":  any(
                p["period_months"] > 20 for p in top_peaks
            ),
            "summary": _fft_summary(top_peaks),
        },
    }


def _fft_summary(peaks: list) -> str:
    if not peaks:
        return "Нет явных циклов"
    dominant = peaks[0]["period_months"]
    parts = []
    for p in peaks[:3]:
        per = p["period_months"]
        if abs(per - 12) < 1.5:
            parts.append(f"годовая сезонность ({per:.0f} мес.)")
        elif 20 <= per <= 40:
            parts.append(f"среднесрочный цикл ({per:.0f} мес.)")
        elif per > 40:
            parts.append(f"долгосрочный цикл ({per:.0f} мес.)")
        else:
            parts.append(f"цикл {per:.0f} мес.")
    return "Обнаружены: " + ", ".join(parts)


# =============================================================================
# 2. АНАЛИЗ ТРЕНДА
# =============================================================================

def get_mann_kendall(series_dict: dict, category: str, channel: str) -> dict:
    """
    Тест Манна-Кендалла на монотонный тренд.

    Непараметрический тест — не требует нормальности, устойчив к выбросам.
    Проверяет гипотезу: есть ли в ряде статистически значимый тренд
    (возрастающий или убывающий).

    КАК ИНТЕРПРЕТИРОВАТЬ:
      - p < 0.05  → тренд статистически значим
      - tau > 0   → возрастающий тренд (продажи растут)
      - tau < 0   → убывающий тренд (продажи падают)
      - |tau| близко к 1 → очень сильный монотонный тренд
      - |tau| близко к 0 → слабый или отсутствующий тренд
      - Sen's slope → на сколько кг в месяц меняются продажи в среднем
    """
    key = (category, channel)
    if key not in series_dict:
        return {}

    s = series_dict[key][TARGET_COL].dropna()
    if len(s) < 12:
        return {"error": "Мало точек для теста"}

    n    = len(s)
    vals = s.values

    # Статистика S (число конкордантных минус дискордантных пар)
    S = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            diff = vals[j] - vals[i]
            S   += np.sign(diff)

    # Дисперсия S
    var_S = n * (n - 1) * (2 * n + 5) / 18

    # Z-статистика
    if S > 0:
        Z = (S - 1) / np.sqrt(var_S)
    elif S < 0:
        Z = (S + 1) / np.sqrt(var_S)
    else:
        Z = 0.0

    p_value = 2 * (1 - stats.norm.cdf(abs(Z)))

    # Tau Кендалла
    n_pairs = n * (n - 1) / 2
    tau     = S / n_pairs

    # Sen's slope — медианный наклон (кг/месяц)
    slopes = []
    for i in range(n - 1):
        for j in range(i + 1, n):
            if j != i:
                slopes.append((vals[j] - vals[i]) / (j - i))
    sens_slope = float(np.median(slopes))

    # Направление тренда
    if p_value < 0.05:
        if tau > 0:
            direction = "Возрастающий (значимый)"
        else:
            direction = "Убывающий (значимый)"
    else:
        direction = "Нет значимого тренда"

    return {
        "category":   category,
        "channel":    channel,
        "S":          round(float(S), 0),
        "Z":          round(float(Z), 4),
        "tau":        round(float(tau), 4),
        "p_value":    round(float(p_value), 4),
        "sens_slope_per_month": round(sens_slope, 2),
        "sens_slope_per_year":  round(sens_slope * 12, 2),
        "significant": bool(p_value < 0.05),
        "direction":   direction,
        "interpretation": (
            f"Тренд {'значим' if p_value < 0.05 else 'незначим'} (p={p_value:.3f}). "
            f"Направление: {direction}. "
            f"Наклон: {sens_slope:.1f} кг/мес. "
            f"({sens_slope * 12:.0f} кг/год)."
        ),
    }


def get_structural_breaks(series_dict: dict, category: str, channel: str,
                           min_size: int = 12) -> dict:
    """
    Поиск точек структурных изломов (Pettitt test + CUSUM).

    Структурный излом — момент когда среднее или тренд ряда резко меняется.
    Типичные причины: COVID-шок, изменение ценовой политики, выход новых SKU.

    Pettitt test: находит ОДНУ наиболее вероятную точку излома.
    CUSUM: показывает накопленные отклонения — визуально видно где ряд
           начал систематически отклоняться от своего среднего.

    КАК ИНТЕРПРЕТИРОВАТЬ:
      - Дата излома + p < 0.05 → статистически значимый структурный сдвиг
      - CUSUM максимум/минимум → визуально соответствует точке излома
      - Если излом в 2020-03 — 2020-06 → COVID-эффект
      - Несколько изломов → ряд прошёл через несколько режимов
      - Разница средних до/после → на сколько кг изменился уровень продаж
    """
    key = (category, channel)
    if key not in series_dict:
        return {}

    s = series_dict[key][TARGET_COL].dropna()
    if len(s) < 2 * min_size:
        return {"error": "Недостаточно точек для анализа изломов"}

    n    = len(s)
    vals = s.values

    # ── Pettitt test ──────────────────────────────────────────────────────────
    # Ищет точку t где сумма знаков разностей максимальна
    U = np.zeros(n)
    for t in range(1, n):
        for i in range(t):
            U[t] += np.sign(vals[t] - vals[i])

    # Статистика K — максимум |U|
    K = float(np.max(np.abs(U)))
    t_break_idx = int(np.argmax(np.abs(U)))

    # p-value аппроксимация
    p_pettitt = 2 * np.exp(-6 * K ** 2 / (n ** 3 + n ** 2))
    p_pettitt = min(float(p_pettitt), 1.0)

    break_date = str(s.index[t_break_idx].date())

    # Средние до и после излома
    mean_before = float(vals[:t_break_idx].mean()) if t_break_idx > 0 else np.nan
    mean_after  = float(vals[t_break_idx:].mean()) if t_break_idx < n else np.nan
    mean_change = mean_after - mean_before if not np.isnan(mean_before) else np.nan
    mean_change_pct = (mean_change / mean_before * 100) if mean_before and mean_before != 0 else np.nan

    # ── CUSUM ─────────────────────────────────────────────────────────────────
    # Накопленная сумма отклонений от среднего
    mean_val = vals.mean()
    cusum    = np.cumsum(vals - mean_val)

    # Нормализованный CUSUM для удобства интерпретации
    cusum_norm = cusum / (vals.std() * np.sqrt(n))

    # Определяем COVID-период
    covid_idx = [
        i for i, d in enumerate(s.index)
        if COVID_START <= str(d.date()) <= COVID_END
    ]
    covid_cusum = {str(s.index[i].date()): round(float(cusum_norm[i]), 4)
                   for i in covid_idx}

    return {
        "category":   category,
        "channel":    channel,
        "dates":      [str(d.date()) for d in s.index],
        "values":     [round(float(v), 2) for v in vals],
        "cusum":      [round(float(v), 4) for v in cusum_norm],
        "break_point": {
            "date":             break_date,
            "index":            t_break_idx,
            "K_statistic":      round(K, 2),
            "p_value":          round(p_pettitt, 4),
            "significant":      bool(p_pettitt < 0.05),
            "mean_before":      round(mean_before, 0) if not np.isnan(mean_before) else None,
            "mean_after":       round(mean_after, 0) if not np.isnan(mean_after) else None,
            "mean_change":      round(mean_change, 0) if not np.isnan(mean_change) else None,
            "mean_change_pct":  round(mean_change_pct, 1) if not np.isnan(mean_change_pct) else None,
        },
        "covid_cusum": covid_cusum,
        "interpretation": (
            f"Наиболее вероятная точка излома: {break_date} "
            f"({'значима' if p_pettitt < 0.05 else 'незначима'}, p={p_pettitt:.3f}). "
            + (f"Среднее изменилось на {mean_change:.0f} кг "
               f"({mean_change_pct:+.1f}%)."
               if not np.isnan(mean_change) else "")
        ),
    }


# =============================================================================
# 3. УГЛУБЛЁННЫЙ АНАЛИЗ СЕЗОННОСТИ
# =============================================================================

def get_seasonal_subseries(series_dict: dict, category: str, channel: str) -> dict:
    """
    Сезонные субграфики (seasonal subseries plot).

    Для каждого месяца года показывает:
      - Среднее значение по всем годам
      - Разброс (min/max/std)
      - Динамику по годам — как менялось значение в этом месяце

    КАК ИНТЕРПРЕТИРОВАТЬ:
      - Высокое среднее в декабре → пик продаж в декабре каждый год
      - Большой разброс в конкретном месяце → нестабильная сезонность
      - Нарастающая линия по годам в одном месяце → рост именно в этот период
      - Провал в 2020 виден как выброс вниз в отдельных месяцах
    """
    key = (category, channel)
    if key not in series_dict:
        return {}

    s = series_dict[key][TARGET_COL].dropna()
    df = s.reset_index()
    df.columns = ["date", "value"]
    df["month"] = df["date"].dt.month
    df["year"]  = df["date"].dt.year

    month_names = {
        1: "Янв", 2: "Фев", 3: "Мар", 4: "Апр",
        5: "Май", 6: "Июн", 7: "Июл", 8: "Авг",
        9: "Сен", 10: "Окт", 11: "Ноя", 12: "Дек",
    }

    months_data = []
    for month in range(1, 13):
        sub = df[df["month"] == month]["value"]
        if len(sub) == 0:
            continue
        months_data.append({
            "month":       month,
            "month_name":  month_names[month],
            "mean":        round(float(sub.mean()), 0),
            "std":         round(float(sub.std()), 0),
            "min":         round(float(sub.min()), 0),
            "max":         round(float(sub.max()), 0),
            "cv":          round(float(sub.std() / sub.mean()), 3) if sub.mean() != 0 else None,
        })

    # По годам для каждого месяца
    yearly = {}
    for year in sorted(df["year"].unique()):
        yearly[int(year)] = {}
        for month in range(1, 13):
            sub = df[(df["year"] == year) & (df["month"] == month)]["value"]
            yearly[int(year)][month] = round(float(sub.values[0]), 0) if len(sub) > 0 else None

    # Самый стабильный и самый нестабильный месяц
    cv_by_month = {m["month"]: m["cv"] for m in months_data if m["cv"]}
    most_stable   = min(cv_by_month, key=cv_by_month.get) if cv_by_month else None
    most_unstable = max(cv_by_month, key=cv_by_month.get) if cv_by_month else None

    return {
        "category":     category,
        "channel":      channel,
        "months":       months_data,
        "yearly":       yearly,
        "peak_month":   max(months_data, key=lambda x: x["mean"])["month_name"] if months_data else None,
        "trough_month": min(months_data, key=lambda x: x["mean"])["month_name"] if months_data else None,
        "most_stable_month":   month_names.get(most_stable),
        "most_unstable_month": month_names.get(most_unstable),
    }


def get_seasonality_test(series_dict: dict, category: str, channel: str) -> dict:
    """
    Тест Краскела-Уоллиса на наличие сезонности.

    Непараметрический аналог однофакторного дисперсионного анализа.
    Проверяет: отличаются ли медианы продаж между месяцами статистически?

    КАК ИНТЕРПРЕТИРОВАТЬ:
      - p < 0.05  → сезонность статистически подтверждена
      - p < 0.01  → сезонность очень значима
      - H-статистика: чем больше, тем сильнее различия между месяцами
      - Eta² ≈ H / (n-1): доля дисперсии объяснённая месяцем (0-1)
        - Eta² > 0.14 → сильная сезонность
        - Eta² 0.06-0.14 → умеренная
        - Eta² < 0.06 → слабая
    """
    key = (category, channel)
    if key not in series_dict:
        return {}

    s = series_dict[key][TARGET_COL].dropna()
    if len(s) < 2 * SEASONAL_PERIOD:
        return {"error": "Мало точек для теста"}

    df = pd.DataFrame({"value": s.values, "month": s.index.month})

    groups = [df[df["month"] == m]["value"].values for m in range(1, 13)
              if len(df[df["month"] == m]) > 0]

    if len(groups) < 2:
        return {"error": "Недостаточно групп"}

    H, p_value = stats.kruskal(*groups)
    n   = len(s)
    eta2 = (H - len(groups) + 1) / (n - len(groups))
    eta2 = max(0.0, float(eta2))

    if p_value < 0.05:
        if eta2 > 0.14:
            strength = "Сильная сезонность"
        elif eta2 > 0.06:
            strength = "Умеренная сезонность"
        else:
            strength = "Слабая сезонность"
    else:
        strength = "Сезонность статистически не подтверждена"

    return {
        "category":   category,
        "channel":    channel,
        "H_statistic": round(float(H), 4),
        "p_value":     round(float(p_value), 4),
        "eta_squared": round(eta2, 4),
        "significant": bool(p_value < 0.05),
        "strength":    strength,
        "interpretation": (
            f"Тест Краскела-Уоллиса: H={H:.2f}, p={p_value:.4f}. "
            f"{strength}. Eta²={eta2:.3f} — месяц объясняет "
            f"{eta2*100:.1f}% дисперсии продаж."
        ),
    }


# =============================================================================
# 4. АНАЛИЗ АНОМАЛИЙ И COVID-ВЛИЯНИЯ
# =============================================================================

def get_anomalies(series_dict: dict, category: str, channel: str,
                   method: str = "iqr", threshold: float = 2.5) -> dict:
    """
    Выявление аномальных точек в ряде.

    Метод 'iqr'  : точка аномальна если выходит за границы
                   Q1 - 1.5*IQR ... Q3 + 1.5*IQR
    Метод 'zscore': точка аномальна если |z-score| > threshold (обычно 2.5-3)
    Метод 'stl'  : аномалия если остаток STL > threshold * MAD(остатков)
                   Лучший метод — учитывает сезонность при поиске выбросов

    КАК ИНТЕРПРЕТИРОВАТЬ:
      - Аномалии в 2020 → COVID-эффект (ожидаемо)
      - Аномалии в другие периоды → промо-акции, перебои поставок, ошибки данных
      - Положительные аномалии → неожиданно высокие продажи
      - Отрицательные аномалии → неожиданно низкие продажи
    """
    from statsmodels.tsa.seasonal import STL

    key = (category, channel)
    if key not in series_dict:
        return {}

    s = series_dict[key][TARGET_COL].dropna()

    anomaly_mask = np.zeros(len(s), dtype=bool)

    if method == "iqr":
        Q1, Q3 = np.percentile(s.values, [25, 75])
        IQR    = Q3 - Q1
        lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        anomaly_mask = (s.values < lower) | (s.values > upper)

    elif method == "zscore":
        z_scores = np.abs(stats.zscore(s.values))
        anomaly_mask = z_scores > threshold

    elif method == "stl":
        if len(s) >= 2 * SEASONAL_PERIOD:
            res  = STL(s, period=SEASONAL_PERIOD, robust=True).fit()
            resid = res.resid.values
            mad  = np.median(np.abs(resid - np.median(resid)))
            anomaly_mask = np.abs(resid - np.median(resid)) > threshold * mad

    anomaly_dates  = [str(s.index[i].date()) for i in range(len(s)) if anomaly_mask[i]]
    anomaly_values = [round(float(s.values[i]), 2) for i in range(len(s)) if anomaly_mask[i]]
    anomaly_types  = [
        "высокий" if s.values[i] > s.mean() else "низкий"
        for i in range(len(s)) if anomaly_mask[i]
    ]

    # Сколько аномалий попало в COVID-период
    covid_anomalies = [
        d for d in anomaly_dates
        if COVID_START <= d <= COVID_END
    ]

    return {
        "category":       category,
        "channel":        channel,
        "method":         method,
        "dates":          [str(d.date()) for d in s.index],
        "values":         [round(float(v), 2) for v in s.values],
        "anomaly_dates":  anomaly_dates,
        "anomaly_values": anomaly_values,
        "anomaly_types":  anomaly_types,
        "n_anomalies":    int(anomaly_mask.sum()),
        "pct_anomalies":  round(float(anomaly_mask.mean() * 100), 1),
        "covid_anomalies": covid_anomalies,
        "interpretation": (
            f"Найдено {anomaly_mask.sum()} аномалий ({anomaly_mask.mean()*100:.1f}% точек). "
            f"Из них {len(covid_anomalies)} в COVID-период (2020-03 — 2020-06)."
        ),
    }


def get_covid_impact(series_dict: dict) -> list[dict]:
    """
    Сравнительный анализ COVID-влияния по всем категориям и каналам.

    Сравнивает продажи в COVID-период (2020-03 — 2020-06) с:
      - аналогичным периодом 2019 года (year-over-year)
      - средним за 12 месяцев до COVID (базовый уровень)

    КАК ИНТЕРПРЕТИРОВАТЬ:
      - drop_yoy < -20% → сильный удар от COVID
      - drop_yoy > 0%   → категория выросла в COVID (товары первой необходимости)
      - recovery_months → за сколько месяцев вернулись к базовому уровню
        - < 3 мес. → быстрое восстановление
        - > 6 мес. → затяжное восстановление
      - Сравнение категорий показывает какие продукты более устойчивы к шокам
    """
    results = []
    for key, df in series_dict.items():
        s = df[TARGET_COL].dropna()

        # COVID-период
        covid_mask = (s.index >= COVID_START) & (s.index <= COVID_END)
        covid_vals = s[covid_mask]
        if len(covid_vals) == 0:
            continue

        covid_mean = float(covid_vals.mean())

        # Аналогичный период 2019
        pre_start = pd.Timestamp(COVID_START) - pd.DateOffset(years=1)
        pre_end   = pd.Timestamp(COVID_END)   - pd.DateOffset(years=1)
        pre_mask  = (s.index >= pre_start) & (s.index <= pre_end)
        pre_vals  = s[pre_mask]
        pre_mean  = float(pre_vals.mean()) if len(pre_vals) > 0 else np.nan

        # Базовый уровень — 12 месяцев до COVID
        base_start = pd.Timestamp(COVID_START) - pd.DateOffset(months=12)
        base_end   = pd.Timestamp(COVID_START) - pd.DateOffset(months=1)
        base_mask  = (s.index >= base_start) & (s.index <= base_end)
        base_vals  = s[base_mask]
        base_mean  = float(base_vals.mean()) if len(base_vals) > 0 else np.nan

        # Падение год к году
        drop_yoy = ((covid_mean - pre_mean) / pre_mean * 100
                    if not np.isnan(pre_mean) and pre_mean != 0 else np.nan)

        # Падение от базового уровня
        drop_base = ((covid_mean - base_mean) / base_mean * 100
                     if not np.isnan(base_mean) and base_mean != 0 else np.nan)

        # Восстановление — когда вернулись к базовому уровню
        recovery_months = None
        if not np.isnan(base_mean):
            post_covid = s[s.index > COVID_END]
            for i, (date, val) in enumerate(post_covid.items()):
                if val >= base_mean * 0.95:
                    recovery_months = i + 1
                    break

        results.append({
            "category":        key[0],
            "channel":         key[1],
            "covid_mean":      round(covid_mean, 0),
            "pre_covid_mean":  round(pre_mean, 0) if not np.isnan(pre_mean) else None,
            "base_mean":       round(base_mean, 0) if not np.isnan(base_mean) else None,
            "drop_yoy_pct":    round(drop_yoy, 1) if not np.isnan(drop_yoy) else None,
            "drop_base_pct":   round(drop_base, 1) if not np.isnan(drop_base) else None,
            "recovery_months": recovery_months,
            "impact_level": (
                "Сильный удар" if drop_yoy is not None and drop_yoy < -20
                else "Умеренный удар" if drop_yoy is not None and drop_yoy < -5
                else "Рост в COVID" if drop_yoy is not None and drop_yoy > 0
                else "Нейтральное влияние"
            ),
        })

    return sorted(results, key=lambda x: x.get("drop_yoy_pct") or 0)


# =============================================================================
# 5. КОРРЕЛЯЦИИ МЕЖДУ РЯДАМИ
# =============================================================================

def get_cross_series_correlation(series_dict: dict,
                                  filter_channel: str = None) -> dict:
    """
    Матрица корреляций между временными рядами.

    Показывает насколько синхронно движутся разные категории продаж
    в рамках одного канала (или по всем каналам).

    КАК ИНТЕРПРЕТИРОВАТЬ:
      - r > 0.8  → очень высокая корреляция — категории движутся почти синхронно
      - r 0.5-0.8 → умеренная корреляция
      - r < 0.3  → слабая корреляция — категории независимы
      - r < 0    → обратная связь — когда одна растёт, другая падает
      - Кластеры высококоррелированных категорий → реагируют на одни факторы
      - Низкая корреляция между категориями → диверсификация риска при агрегации
    """
    # Фильтруем по каналу если указан
    keys = [k for k in series_dict.keys()
            if filter_channel is None or k[1] == filter_channel]

    if len(keys) < 2:
        return {"error": "Нужно минимум 2 ряда для корреляционного анализа"}

    # Собираем все ряды в одну таблицу по общим датам
    series_df = pd.DataFrame()
    for key in keys:
        s = series_dict[key][TARGET_COL].dropna()
        label = f"{key[0]}"[:20]  # Только категория для компактности
        series_df[label] = s

    series_df = series_df.dropna()
    if len(series_df) < 12:
        return {"error": "Мало общих наблюдений для корреляционного анализа"}

    corr_matrix = series_df.corr(method="pearson")
    labels      = list(corr_matrix.columns)

    # Топ коррелированных пар
    pairs = []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            r = corr_matrix.iloc[i, j]
            pairs.append({
                "series_a": labels[i],
                "series_b": labels[j],
                "r":        round(float(r), 4),
            })
    pairs_sorted = sorted(pairs, key=lambda x: abs(x["r"]), reverse=True)

    return {
        "channel":  filter_channel or "all",
        "labels":   labels,
        "matrix":   [[round(float(corr_matrix.iloc[i, j]), 4)
                      for j in range(len(labels))]
                     for i in range(len(labels))],
        "top_pairs":       pairs_sorted[:5],
        "bottom_pairs":    sorted(pairs, key=lambda x: x["r"])[:3],
        "n_series":        len(labels),
        "mean_correlation": round(float(np.array([p["r"] for p in pairs]).mean()), 4),
    }


def get_cross_correlation(series_dict: dict,
                           category_a: str, channel_a: str,
                           category_b: str, channel_b: str,
                           n_lags: int = 12) -> dict:
    """
    Кросс-корреляция (CCF) между двумя рядами.

    Показывает как коррелируют ряды со сдвигом — опережает ли один ряд другой.

    КАК ИНТЕРПРЕТИРОВАТЬ:
      - Пик на лаге 0  → синхронное движение
      - Пик на лаге +k → ряд A опережает ряд B на k месяцев
      - Пик на лаге -k → ряд B опережает ряд A на k месяцев
      - Это полезно для определения причинно-следственных связей
        (например, рост одной категории предсказывает рост другой)
    """
    key_a = (category_a, channel_a)
    key_b = (category_b, channel_b)

    if key_a not in series_dict or key_b not in series_dict:
        return {"error": "Один или оба ряда не найдены"}

    s_a = series_dict[key_a][TARGET_COL].dropna()
    s_b = series_dict[key_b][TARGET_COL].dropna()

    common = s_a.index.intersection(s_b.index)
    if len(common) < 20:
        return {"error": "Мало общих наблюдений"}

    a_vals = s_a[common].values
    b_vals = s_b[common].values

    # Нормализуем
    a_norm = (a_vals - a_vals.mean()) / a_vals.std()
    b_norm = (b_vals - b_vals.mean()) / b_vals.std()

    ccf_vals = ccf(a_norm, b_norm, nlags=n_lags, alpha=None)
    lags     = list(range(-n_lags, n_lags + 1))

    # CCF в обоих направлениях
    ccf_ab = ccf(a_norm, b_norm, nlags=n_lags, alpha=None).tolist()
    ccf_ba = ccf(b_norm, a_norm, nlags=n_lags, alpha=None).tolist()

    cb = 1.96 / np.sqrt(len(common))

    # Находим лаг с максимальной корреляцией
    max_lag_idx = int(np.argmax(np.abs(ccf_ab)))
    max_lag     = max_lag_idx  # лаг 0 = индекс 0
    max_corr    = float(ccf_ab[max_lag_idx])

    return {
        "series_a":   f"{category_a} | {channel_a}",
        "series_b":   f"{category_b} | {channel_b}",
        "lags":       list(range(len(ccf_ab))),
        "ccf":        [round(v, 4) for v in ccf_ab],
        "conf_bound": round(cb, 4),
        "max_lag":    max_lag,
        "max_corr":   round(max_corr, 4),
        "interpretation": (
            f"Максимальная корреляция {max_corr:.3f} на лаге {max_lag}. "
            + (f"Ряд A опережает ряд B на {max_lag} мес."
               if max_lag > 0 else
               f"Ряд B опережает ряд A на {abs(max_lag)} мес."
               if max_lag < 0 else
               "Ряды движутся синхронно.")
        ),
    }


# =============================================================================
# Сводный анализ по всем рядам (для дашборда)
# =============================================================================

def get_full_analytics_summary(series_dict: dict) -> dict:
    """
    Сводная таблица по всем рядам — запускается один раз,
    используется для общего обзора на главной странице аналитики.
    """
    rows = []
    for key, df in series_dict.items():
        s = df[TARGET_COL].dropna()

        # Mann-Kendall (упрощённо через scipy)
        mk = get_mann_kendall(series_dict, key[0], key[1])

        # Kruskal-Wallis
        kw = get_seasonality_test(series_dict, key[0], key[1])

        rows.append({
            "category":         key[0],
            "channel":          key[1],
            "n_obs":            int(len(s)),
            "mean":             round(float(s.mean()), 0),
            "trend_direction":  mk.get("direction", "—") if "error" not in mk else "—",
            "trend_significant": mk.get("significant", False) if "error" not in mk else False,
            "sens_slope_year":  mk.get("sens_slope_per_year") if "error" not in mk else None,
            "seasonality":      kw.get("strength", "—") if "error" not in kw else "—",
            "seasonality_significant": kw.get("significant", False) if "error" not in kw else False,
            "eta_squared":      kw.get("eta_squared") if "error" not in kw else None,
        })

    return {"summary": rows, "n_series": len(rows)}