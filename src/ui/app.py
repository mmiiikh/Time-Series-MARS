"""
Streamlit-интерфейс для сервиса прогнозирования продаж Mars.
Запуск: streamlit run src/ui/app.py
"""
import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from io import BytesIO

API_URL = "http://localhost:8000"

st.set_page_config(page_title="Mars Sales Forecasting",layout="wide")


@st.cache_data(ttl=300)
def fetch_series() -> list[dict]:
    try:
        r = requests.get(f"{API_URL}/series", timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"Не удалось подключиться к API: {e}")
        return []


@st.cache_data(ttl=300)
def fetch_history(category: str, channel: str) -> dict:
    try:
        r = requests.get(f"{API_URL}/series/{category}/{channel}/history", timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"Ошибка получения данных: {e}")
        return {}


@st.cache_data(ttl=300)
def fetch_stl(category: str, channel: str) -> dict:
    try:
        r = requests.get(f"{API_URL}/series/{category}/{channel}/stl", timeout=30)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"Ошибка STL-декомпозиции: {e}")
        return {}


@st.cache_data(ttl=300)
def fetch_stationarity(category: str, channel: str) -> dict:
    try:
        r = requests.get(f"{API_URL}/series/{category}/{channel}/stationarity", timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"Ошибка тестов стационарности: {e}")
        return {}


@st.cache_data(ttl=300)
def fetch_acf_pacf(category: str, channel: str, diff: str = "original") -> dict:
    try:
        r = requests.get(
            f"{API_URL}/series/{category}/{channel}/acf_pacf",
            params={"diff": diff}, timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"Ошибка ACF/PACF: {e}")
        return {}


@st.cache_data(ttl=300)
def fetch_model_info(category: str, channel: str) -> dict:
    try:
        r = requests.get(f"{API_URL}/models/info/{category}/{channel}", timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception:
        return {}


def fetch_forecast(category: str, channel: str,
                   horizon: int, exog_file=None) -> dict:
    try:
        if exog_file is not None:
            r = requests.post(
                f"{API_URL}/forecast/upload",
                params={"category": category, "channel": channel,
                        "horizon": horizon, "model_class": "econometric"},
                files={"file": (exog_file.name, exog_file.getvalue(),
                                "application/octet-stream")},
                timeout=60,
            )
        else:
            r = requests.post(
                f"{API_URL}/forecast",
                json={"category": category, "channel": channel,
                      "horizon": horizon, "model_class": "econometric"},
                timeout=60,
            )
        r.raise_for_status()
        return r.json()
    except requests.HTTPError as e:
        st.error(f"Ошибка API ({r.status_code}): {r.json().get('detail', str(e))}")
        return {}
    except Exception as e:
        st.error(f"Ошибка прогноза: {e}")
        return {}


def build_history_chart(histories: list[dict], title: str = "") -> go.Figure:
    fig = go.Figure()
    for h in histories:
        label = f"{h['category']} | {h['channel']}"
        fig.add_trace(go.Scatter(
            x=h["dates"], y=h["values"],
            mode="lines", name=label,
            hovertemplate="%{x}<br>%{y:,.0f} кг<extra>" + label + "</extra>"))
    fig.update_layout(
        title=title or "Исторические данные",
        xaxis_title="Дата", yaxis_title="Объём продаж, кг",
        hovermode="x unified", legend=dict(orientation="h", y=-0.2),
        height=420)
    return fig


def build_stl_chart(stl: dict) -> go.Figure:
    """4-компонентный STL-график."""
    fig = go.Figure()
    components = [
        ("observed",  "Факт",        "#1f77b4"),
        ("trend",     "Тренд",       "#d62728"),
        ("seasonal",  "Сезонность",  "#ff7f0e"),
        ("resid",     "Остаток",     "#9467bd")]
    # Subplot вручную через traces с разным y-axis
    fig = go.Figure()
    from plotly.subplots import make_subplots
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                        subplot_titles=["Факт", "Тренд", "Сезонность", "Остаток"],
                        vertical_spacing=0.06)
    for row, (key, label, color) in enumerate(components, 1):
        fig.add_trace(go.Scatter(
            x=stl["dates"], y=stl[key],
            mode="lines", name=label,
            line=dict(color=color, width=1.5),
            hovertemplate="%{x}<br>%{y:,.0f}<extra></extra>",
        ), row=row, col=1)

    fig.update_layout(
        height=700, showlegend=False,
        title=f"STL-декомпозиция: {stl['category']} | {stl['channel']}",
    )
    return fig


def build_acf_pacf_chart(data: dict) -> go.Figure:
    """ACF и PACF на одном графике (два subplot)."""
    from plotly.subplots import make_subplots
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=["ACF", "PACF"])
    cb  = data["conf_bound"]

    for col, (vals, name) in enumerate(
        [(data["acf"], "ACF"), (data["pacf"], "PACF")], 1
    ):
        lags = data["lags"]
        # Вертикальные линии (стержни)
        for lag, val in zip(lags, vals):
            fig.add_trace(go.Scatter(
                x=[lag, lag], y=[0, val],
                mode="lines",
                line=dict(color="#1f77b4", width=1.5),
                showlegend=False,
                hovertemplate=f"Lag {lag}: {val:.3f}<extra></extra>",
            ), row=1, col=col)

        # Доверительные границы
        fig.add_hline(y=cb,  line_dash="dash", line_color="red",
                      annotation_text="95% ДИ", row=1, col=col)
        fig.add_hline(y=-cb, line_dash="dash", line_color="red", row=1, col=col)
        fig.add_hline(y=0,   line_color="black", line_width=0.8, row=1, col=col)

        # Сезонные лаги — вертикальные пунктиры
        for sl in data.get("seasonal_lags", []):
            fig.add_vline(x=sl, line_dash="dot", line_color="grey",
                          opacity=0.6, row=1, col=col)

    fig.update_layout(height=380, showlegend=False,
                      title=f"ACF / PACF — {data['diff']}")
    return fig


def build_forecast_chart(history: dict, forecast: dict) -> go.Figure:
    """Исторический ряд + прогноз с доверительным интервалом."""
    fig = go.Figure()

    # История
    fig.add_trace(go.Scatter(
        x=history["dates"], y=history["values"],
        mode="lines", name="История",
        line=dict(color="#1f77b4", width=2),
        hovertemplate="%{x}<br>%{y:,.0f} кг<extra>История</extra>",
    ))

    fc_points = forecast["forecast"]
    dates  = [p["date"]     for p in fc_points]
    values = [p["forecast"] for p in fc_points]
    lower  = [p["lower_80"] for p in fc_points]
    upper  = [p["upper_80"] for p in fc_points]

    # Доверительный интервал
    fig.add_trace(go.Scatter(
        x=dates + dates[::-1],
        y=upper + lower[::-1],
        fill="toself", fillcolor="rgba(214,39,40,0.15)",
        line=dict(color="rgba(255,255,255,0)"),
        name="80% ДИ", hoverinfo="skip",
    ))

    # Прогноз
    fig.add_trace(go.Scatter(
        x=dates, y=values,
        mode="lines+markers", name="Прогноз",
        line=dict(color="#d62728", width=2, dash="dash"),
        marker=dict(size=5),
        hovertemplate="%{x}<br>%{y:,.0f} кг<extra>Прогноз</extra>",
    ))

    # Вертикальная граница история/прогноз
    if history["dates"]:
        fig.add_vline(x=history["dates"][-1], line_dash="dot",
                      line_color="grey", opacity=0.7)

    fig.update_layout(
        title=f"Прогноз: {forecast['category']} | {forecast['channel']}",
        xaxis_title="Дата", yaxis_title="Объём продаж, кг",
        hovermode="x unified", legend=dict(orientation="h", y=-0.2),
        height=450,
    )
    return fig


def build_aggregate_forecast_chart(histories: list[dict],
                                    forecasts: list[dict],
                                    label: str) -> go.Figure:
    """Суммарный исторический ряд + суммарный прогноз по нескольким рядам."""
    fig = go.Figure()

    # Суммируем историю по общим датам
    hist_df = pd.DataFrame()
    for h in histories:
        tmp = pd.DataFrame({"date": h["dates"], "value": h["values"]})
        hist_df = pd.concat([hist_df, tmp])
    hist_agg = hist_df.groupby("date")["value"].sum().reset_index()

    fig.add_trace(go.Scatter(
        x=hist_agg["date"], y=hist_agg["value"],
        mode="lines", name="История (сумма)",
        line=dict(color="#1f77b4", width=2),
        hovertemplate="%{x}<br>%{y:,.0f} кг<extra>История</extra>",
    ))

    # Суммируем прогноз
    fc_df = pd.DataFrame()
    for fc in forecasts:
        tmp = pd.DataFrame(fc["forecast"])
        fc_df = pd.concat([fc_df, tmp])
    fc_agg = fc_df.groupby("date").agg(
        forecast=("forecast", "sum"),
        lower_80=("lower_80", "sum"),
        upper_80=("upper_80", "sum"),
    ).reset_index()

    fig.add_trace(go.Scatter(
        x=fc_agg["date"].tolist() + fc_agg["date"].tolist()[::-1],
        y=fc_agg["upper_80"].tolist() + fc_agg["lower_80"].tolist()[::-1],
        fill="toself", fillcolor="rgba(214,39,40,0.15)",
        line=dict(color="rgba(255,255,255,0)"),
        name="80% ДИ", hoverinfo="skip",
    ))

    fig.add_trace(go.Scatter(
        x=fc_agg["date"], y=fc_agg["forecast"],
        mode="lines+markers", name="Прогноз (сумма)",
        line=dict(color="#d62728", width=2, dash="dash"),
        marker=dict(size=5),
        hovertemplate="%{x}<br>%{y:,.0f} кг<extra>Прогноз</extra>",
    ))

    if not hist_agg.empty:
        fig.add_vline(x=hist_agg["date"].iloc[-1], line_dash="dot",
                      line_color="grey", opacity=0.7)

    fig.update_layout(
        title=f"Суммарный прогноз: {label}",
        xaxis_title="Дата", yaxis_title="Объём продаж, кг",
        hovermode="x unified", legend=dict(orientation="h", y=-0.2),
        height=450,
    )
    return fig


# =============================================================================
# Fetch-функции для новой аналитики
# =============================================================================

@st.cache_data(ttl=300)
def fetch_hp_filter(category: str, channel: str) -> dict:
    try:
        r = requests.get(f"{API_URL}/series/{category}/{channel}/hp_filter", timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception:
        return {}


@st.cache_data(ttl=300)
def fetch_fft(category: str, channel: str) -> dict:
    try:
        r = requests.get(f"{API_URL}/series/{category}/{channel}/fft", timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception:
        return {}


@st.cache_data(ttl=300)
def fetch_mann_kendall(category: str, channel: str) -> dict:
    try:
        r = requests.get(f"{API_URL}/series/{category}/{channel}/mann_kendall", timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception:
        return {}


@st.cache_data(ttl=300)
def fetch_structural_breaks(category: str, channel: str) -> dict:
    try:
        r = requests.get(f"{API_URL}/series/{category}/{channel}/structural_breaks", timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception:
        return {}


@st.cache_data(ttl=300)
def fetch_seasonal_subseries(category: str, channel: str) -> dict:
    try:
        r = requests.get(f"{API_URL}/series/{category}/{channel}/seasonal_subseries", timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception:
        return {}


@st.cache_data(ttl=300)
def fetch_seasonality_test(category: str, channel: str) -> dict:
    try:
        r = requests.get(f"{API_URL}/series/{category}/{channel}/seasonality_test", timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception:
        return {}


@st.cache_data(ttl=300)
def fetch_anomalies(category: str, channel: str, method: str = "stl") -> dict:
    try:
        r = requests.get(
            f"{API_URL}/series/{category}/{channel}/anomalies",
            params={"method": method}, timeout=15
        )
        r.raise_for_status()
        return r.json()
    except Exception:
        return {}


@st.cache_data(ttl=300)
def fetch_covid_impact(channel: str = None) -> list:
    try:
        params = {"channel": channel} if channel else {}
        r = requests.get(f"{API_URL}/analytics/covid_impact", params=params, timeout=20)
        r.raise_for_status()
        return r.json()
    except Exception:
        return []


@st.cache_data(ttl=300)
def fetch_correlation(channel: str = None) -> dict:
    try:
        params = {"channel": channel} if channel else {}
        r = requests.get(f"{API_URL}/analytics/correlation", params=params, timeout=20)
        r.raise_for_status()
        return r.json()
    except Exception:
        return {}


@st.cache_data(ttl=300)
def fetch_analytics_summary() -> dict:
    try:
        r = requests.get(f"{API_URL}/analytics/summary", timeout=30)
        r.raise_for_status()
        return r.json()
    except Exception:
        return {}


# =============================================================================
# Страница: Аналитика
# =============================================================================

def page_analytics(series_list: list[dict]):
    st.title("Аналитика временных рядов")

    categories = sorted({s["category"] for s in series_list})
    channels   = sorted({s["channel"]  for s in series_list})

    # ── Фильтры ──────────────────────────────────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        selected_cats = st.multiselect(
            "Категории", categories, default=[categories[0]],
            placeholder="Выберите категории...",
        )
    with col2:
        selected_chs = st.multiselect(
            "Каналы", channels, default=[channels[0]],
            placeholder="Выберите каналы...",
        )

    if not selected_cats or not selected_chs:
        st.info("Выберите хотя бы одну категорию и один канал.")
        return

    available_pairs = [
        s for s in series_list
        if s["category"] in selected_cats and s["channel"] in selected_chs
    ]
    if not available_pairs:
        st.warning("Нет данных для выбранной комбинации.")
        return

    # ── Исторические графики ─────────────────────────────────────────────────
    st.subheader("Исторические данные")
    histories = [h for p in available_pairs
                 if (h := fetch_history(p["category"], p["channel"]))]
    if histories:
        st.plotly_chart(
            build_history_chart(histories, "Объём продаж по выбранным рядам"),
            use_container_width=True,
        )

    with st.expander("Базовые статистики"):
        stats_df = pd.DataFrame([{
            "Категория":  s["category"],
            "Канал":      s["channel"],
            "Наблюдений": s["n_obs"],
            "С":          s["date_from"],
            "По":         s["date_to"],
        } for s in available_pairs])
        st.dataframe(stats_df, use_container_width=True, hide_index=True)

    st.divider()

    # ── Выбор ряда для детального анализа ────────────────────────────────────
    st.subheader("Детальный анализ ряда")
    pair_labels   = [f"{s['category']} | {s['channel']}" for s in available_pairs]
    selected_label = st.selectbox("Ряд для анализа", pair_labels)
    selected_pair  = available_pairs[pair_labels.index(selected_label)]
    cat, ch = selected_pair["category"], selected_pair["channel"]

    # 8 вкладок детального анализа
    (tab_stl, tab_cycle, tab_fft,
     tab_trend, tab_seas, tab_stat,
     tab_acf, tab_anom) = st.tabs([
        "STL-декомпозиция",
        "Цикличность (HP)",
        "Спектральный анализ",
        "Тренд",
        "Сезонность",
        "Стационарность",
        "ACF / PACF",
        "Аномалии",
    ])

    # ── 1. STL ───────────────────────────────────────────────────────────────
    with tab_stl:
        with st.spinner("Считаем STL-декомпозицию..."):
            stl = fetch_stl(cat, ch)
        if stl and "error" not in stl:
            m = stl["metrics"]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Сила сезонности Fs", f"{m['Fs_seasonal']:.3f}",
                      help="0 = нет сезонности, 1 = идеальная сезонность")
            c2.metric("Сила тренда Ft",     f"{m['Ft_trend']:.3f}",
                      help="0 = нет тренда, 1 = чистый тренд")
            c3.metric("R²",                 f"{m['R2']:.3f}",
                      help="Доля дисперсии объяснённая трендом + сезонностью")
            c4.metric("Амплитуда сезонности", f"{m['seas_amplitude']:,.0f} кг",
                      help="Разница между max и min сезонной компоненты")
            st.plotly_chart(build_stl_chart(stl), use_container_width=True)
        elif stl and "error" in stl:
            st.warning(stl["error"])

    # ── 2. HP-фильтр (цикличность) ───────────────────────────────────────────
    with tab_cycle:
        with st.spinner("Применяем HP-фильтр..."):
            hp = fetch_hp_filter(cat, ch)
        if hp and "error" not in hp:
            m = hp["metrics"]
            interp = hp.get("interpretation", {})

            c1, c2, c3 = st.columns(3)
            c1.metric("Фаза цикла",       interp.get("current_phase", "—"))
            c2.metric("Амплитуда цикла",  f"{m['cycle_amplitude']:,.0f} кг",
                      help="Разница между max и min циклической компоненты")
            c3.metric("Средняя длина цикла",
                      f"{m['avg_cycle_months']} мес." if m.get("avg_cycle_months") else "—")

            st.caption(f"**{interp.get('cycle_strength', '')}** — "
                       f"{m['pct_above_trend']:.0f}% времени продажи выше тренда")

            from plotly.subplots import make_subplots
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                subplot_titles=["Факт и тренд (HP)", "Циклическая компонента"],
                                vertical_spacing=0.1)
            fig.add_trace(go.Scatter(x=hp["dates"], y=hp["observed"],
                                     name="Факт", line=dict(color="#aec6e8", width=1.5)),
                          row=1, col=1)
            fig.add_trace(go.Scatter(x=hp["dates"], y=hp["trend"],
                                     name="Тренд", line=dict(color="#1f77b4", width=2.5)),
                          row=1, col=1)
            fig.add_trace(go.Bar(x=hp["dates"], y=hp["cycle"],
                                 name="Цикл",
                                 marker_color=["#d62728" if v < 0 else "#2ca02c"
                                               for v in hp["cycle"]]),
                          row=2, col=1)
            fig.add_hline(y=0, line_color="black", line_width=0.8, row=2, col=1)
            fig.update_layout(height=500, hovermode="x unified",
                              title=f"HP-фильтр: {cat} | {ch}")
            st.plotly_chart(fig, use_container_width=True)

            st.info("**Как читать:** зелёные столбцы — продажи выше долгосрочного тренда "
                    "(фаза подъёма), красные — ниже (фаза спада). "
                    "Провал в 2020 — COVID-шок.")
        elif hp and "error" in hp:
            st.warning(hp["error"])

    # ── 3. FFT (спектральный анализ) ─────────────────────────────────────────
    with tab_fft:
        with st.spinner("Проводим спектральный анализ..."):
            fft_data = fetch_fft(cat, ch)
        if fft_data and "error" not in fft_data:
            interp = fft_data.get("interpretation", {})

            st.caption(f"**{interp.get('summary', '')}**")

            c1, c2 = st.columns(2)
            with c1:
                dom = interp.get("dominant_period")
                st.metric("Доминирующий период", f"{dom:.0f} мес." if dom else "—")
                st.metric("Длинные циклы (>20 мес.)",
                          "Есть ✔" if interp.get("has_long_cycle") else "Не обнаружены")
            with c2:
                st.markdown("**Топ периодов:**")
                for peak in fft_data.get("top_peaks", [])[:5]:
                    bar = "█" * int(peak["relative_power"] / 10)
                    st.markdown(
                        f"`{peak['period_months']:5.1f} мес.`  {bar}  "
                        f"{peak['relative_power']:.0f}%"
                    )

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=fft_data["periods"], y=fft_data["power"],
                mode="lines", fill="tozeroy",
                line=dict(color="#1f77b4"),
                hovertemplate="Период: %{x:.1f} мес.<br>Мощность: %{y:.1f}%<extra></extra>",
            ))
            for peak in fft_data.get("top_peaks", [])[:3]:
                fig.add_vline(x=peak["period_months"], line_dash="dash",
                              line_color="#d62728", opacity=0.7,
                              annotation_text=f"{peak['period_months']:.0f}м",
                              annotation_position="top")
            fig.update_layout(
                title="Спектр мощности (FFT)",
                xaxis_title="Период, месяцев",
                yaxis_title="Относительная мощность, %",
                xaxis=dict(range=[0, min(60, max(fft_data["periods"]))]),
                height=380,
            )
            st.plotly_chart(fig, use_container_width=True)

            st.info("**Как читать:** пик на 12 мес. = годовая сезонность (ожидаемо). "
                    "Пик на 24-36 мес. = среднесрочный цикл восстановления. "
                    "Пик >48 мес. = долгосрочный тренд-цикл.")
        elif fft_data and "error" in fft_data:
            st.warning(fft_data["error"])

    # ── 4. Тренд (Манн-Кендалл + структурные изломы) ─────────────────────────
    with tab_trend:
        col_mk, col_br = st.columns(2)

        with col_mk:
            st.markdown("#### Тест Манна-Кендалла")
            with st.spinner("..."):
                mk = fetch_mann_kendall(cat, ch)
            if mk and "error" not in mk:
                c1, c2 = st.columns(2)
                c1.metric("Направление", mk["direction"].split(" ")[0])
                c2.metric("p-value", f"{mk['p_value']:.4f}",
                          delta="значим" if mk["significant"] else "незначим",
                          delta_color="normal" if mk["significant"] else "off")
                c1.metric("Tau Кендалла", f"{mk['tau']:.4f}",
                          help="-1 = убывающий, +1 = возрастающий")
                c2.metric("Sen's slope", f"{mk['sens_slope_per_year']:+,.0f} кг/год",
                          help="Медианное изменение продаж в год")
                st.caption(mk.get("interpretation", ""))
            elif mk and "error" in mk:
                st.warning(mk["error"])

        with col_br:
            st.markdown("#### Структурные изломы (Pettitt + CUSUM)")
            with st.spinner("..."):
                sb = fetch_structural_breaks(cat, ch)
            if sb and "error" not in sb:
                bp = sb["break_point"]
                c1, c2 = st.columns(2)
                c1.metric("Точка излома", bp["date"])
                c2.metric("p-value",      f"{bp['p_value']:.4f}",
                          delta="значим" if bp["significant"] else "незначим",
                          delta_color="normal" if bp["significant"] else "off")
                if bp.get("mean_change") is not None:
                    c1.metric("Изменение среднего",
                              f"{bp['mean_change']:+,.0f} кг",
                              delta=f"{bp['mean_change_pct']:+.1f}%")
                st.caption(sb.get("interpretation", ""))

        # CUSUM-график
        if sb and "error" not in sb:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=sb["dates"], y=sb["cusum"],
                mode="lines", name="CUSUM",
                line=dict(color="#1f77b4", width=2),
                hovertemplate="%{x}<br>CUSUM: %{y:.3f}<extra></extra>",
            ))
            fig.add_hline(y=0, line_color="black", line_width=0.8)
            # Точка излома
            bp = sb["break_point"]
            fig.add_shape(
                type="line",
                x0=bp["date"], x1=bp["date"],
                y0=0, y1=1,
                yref="paper",
                line=dict(color="#d62728", dash="dash", width=1.5),
            )
            fig.add_annotation(
                x=bp["date"], y=1,
                yref="paper",
                text=f"Излом: {bp['date']}",
                showarrow=False,
                xanchor="left",
                font=dict(color="#d62728"),
            )
            fig.update_layout(
                title="CUSUM — накопленные отклонения от среднего",
                xaxis_title="Дата",
                yaxis_title="Нормализованный CUSUM",
                height=320,
            )
            st.plotly_chart(fig, use_container_width=True)
            st.info("**Как читать:** CUSUM растёт когда продажи систематически выше среднего, "
                    "падает — когда ниже. Резкий перелом = структурный сдвиг.")

    # ── 5. Сезонность ─────────────────────────────────────────────────────────
    with tab_seas:
        col_kw, col_sub = st.columns([1, 2])

        with col_kw:
            st.markdown("#### Тест Краскела-Уоллиса")
            with st.spinner("..."):
                kw = fetch_seasonality_test(cat, ch)
            if kw and "error" not in kw:
                st.metric("Вывод", kw["strength"])
                st.metric("p-value",      f"{kw['p_value']:.4f}",
                          delta="значима" if kw["significant"] else "незначима",
                          delta_color="normal" if kw["significant"] else "off")
                st.metric("Eta²", f"{kw['eta_squared']:.3f}",
                          help="Доля дисперсии объяснённая месяцем")
                st.caption(kw.get("interpretation", ""))

        with col_sub:
            st.markdown("#### Сезонные субграфики")
            with st.spinner("..."):
                ss = fetch_seasonal_subseries(cat, ch)
            if ss and ss.get("months"):
                months = ss["months"]

                c1, c2 = st.columns(2)
                c1.metric("Пик продаж",   ss.get("peak_month", "—"))
                c2.metric("Спад продаж",  ss.get("trough_month", "—"))
                c1.metric("Стабильный месяц",   ss.get("most_stable_month", "—"),
                          help="Наименьший коэффициент вариации")
                c2.metric("Нестабильный месяц", ss.get("most_unstable_month", "—"),
                          help="Наибольший коэффициент вариации")

                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=[m["month_name"] for m in months],
                    y=[m["mean"]       for m in months],
                    error_y=dict(type="data",
                                 array=[m["std"] for m in months],
                                 visible=True),
                    marker_color="#1f77b4",
                    hovertemplate="%{x}<br>Среднее: %{y:,.0f} кг<extra></extra>",
                    name="Среднее по месяцу",
                ))
                fig.update_layout(
                    title="Средние продажи по месяцам (±std)",
                    xaxis_title="Месяц",
                    yaxis_title="Объём, кг",
                    height=320,
                )
                st.plotly_chart(fig, use_container_width=True)

        # По годам
        if ss and ss.get("yearly"):
            st.markdown("#### Динамика по годам")
            yearly = ss["yearly"]
            month_names = ["Янв","Фев","Мар","Апр","Май","Июн",
                           "Июл","Авг","Сен","Окт","Ноя","Дек"]
            fig = go.Figure()
            for year, vals in sorted(yearly.items()):
                y_vals = [vals.get(m) for m in range(1, 13)]
                fig.add_trace(go.Scatter(
                    x=month_names, y=y_vals,
                    mode="lines+markers", name=str(year),
                    hovertemplate=f"{year} %{{x}}: %{{y:,.0f}} кг<extra></extra>",
                ))
            fig.update_layout(
                title="Сезонный паттерн по годам",
                xaxis_title="Месяц", yaxis_title="Объём, кг",
                hovermode="x unified",
                height=350,
            )
            st.plotly_chart(fig, use_container_width=True)
            st.info("**Как читать:** провал 2020 в отдельных месяцах — COVID-эффект. "
                    "Сближение линий — стабильный сезонный паттерн. "
                    "Расхождение — сезонность меняется со временем.")

    # ── 6. Стационарность ────────────────────────────────────────────────────
    with tab_stat:
        with st.spinner("Запускаем тесты стационарности..."):
            stat = fetch_stationarity(cat, ch)
        if stat and "tests" in stat:
            diff_labels = {
                "original": "Исходный ряд",
                "d=1":      "Первая разность (d=1)",
                "D=1":      "Сезонная разность (D=1)",
                "d=1+D=1":  "Обе разности (d=1, D=1)",
            }
            rows = []
            for diff_key, result in stat["tests"].items():
                rows.append({
                    "Уровень":      diff_labels.get(diff_key, diff_key),
                    "ADF stat":     result["adf_stat"],
                    "ADF p-value":  result["adf_p"],
                    "KPSS stat":    result["kpss_stat"],
                    "KPSS p-value": result["kpss_p"],
                    "Вывод":        result["conclusion"],
                    "✔/✘":          "✔" if result["is_stationary"] else "✘",
                })
            df_stat = pd.DataFrame(rows)
            st.dataframe(
                df_stat.style.map(
                    lambda v: "color: green" if v == "✔" else
                              "color: red"   if v == "✘" else "",
                    subset=["✔/✘"],
                ),
                use_container_width=True, hide_index=True,
            )
            st.info("**ADF:** p < 0.05 → ряд стационарен. "
                    "**KPSS:** p > 0.05 → ряд стационарен. "
                    "Оба должны подтвердить одно и то же.")

    # ── 7. ACF / PACF ────────────────────────────────────────────────────────
    with tab_acf:
        diff_option = st.selectbox(
            "Дифференцирование",
            ["original", "d=1", "D=1", "d=1+D=1"],
            format_func=lambda x: {
                "original": "Исходный ряд",
                "d=1":      "d=1 (первая разность)",
                "D=1":      "D=1 (сезонная разность)",
                "d=1+D=1":  "d=1 + D=1",
            }.get(x, x),
        )
        with st.spinner("Считаем ACF/PACF..."):
            acf_data = fetch_acf_pacf(cat, ch, diff=diff_option)
        if acf_data and "error" not in acf_data:
            st.plotly_chart(build_acf_pacf_chart(acf_data), use_container_width=True)
            st.caption(f"Доверительная граница ±{acf_data['conf_bound']:.3f} (95%)")
            st.info("**Как читать:** хвост ACF убывает медленно → AR-компонент. "
                    "Обрыв PACF на лаге k → порядок p=k. "
                    "Пики на лагах 12, 24 → сезонность.")
        elif acf_data and "error" in acf_data:
            st.warning(acf_data["error"])

    # ── 8. Аномалии ──────────────────────────────────────────────────────────
    with tab_anom:
        method = st.selectbox(
            "Метод выявления аномалий",
            ["stl", "iqr", "zscore"],
            format_func=lambda x: {
                "stl":    "STL-остатки (рекомендуется)",
                "iqr":    "IQR (межквартильный размах)",
                "zscore": "Z-score",
            }.get(x, x),
        )
        with st.spinner("Ищем аномалии..."):
            anom = fetch_anomalies(cat, ch, method=method)

        if anom:
            c1, c2, c3 = st.columns(3)
            c1.metric("Аномалий найдено",    anom["n_anomalies"])
            c2.metric("Доля аномалий",       f"{anom['pct_anomalies']:.1f}%")
            c3.metric("Из них в COVID-период", len(anom.get("covid_anomalies", [])))
            st.caption(anom.get("interpretation", ""))

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=anom["dates"], y=anom["values"],
                mode="lines", name="Факт",
                line=dict(color="#1f77b4", width=1.5),
                hovertemplate="%{x}<br>%{y:,.0f} кг<extra></extra>",
            ))
            if anom["anomaly_dates"]:
                colors = ["#d62728" if t == "низкий" else "#ff7f0e"
                          for t in anom["anomaly_types"]]
                fig.add_trace(go.Scatter(
                    x=anom["anomaly_dates"],
                    y=anom["anomaly_values"],
                    mode="markers", name="Аномалия",
                    marker=dict(color=colors, size=10, symbol="x"),
                    hovertemplate="%{x}<br>%{y:,.0f} кг<extra>Аномалия</extra>",
                ))
            fig.update_layout(
                title=f"Аномалии ({method})",
                xaxis_title="Дата", yaxis_title="Объём, кг",
                height=380,
            )
            st.plotly_chart(fig, use_container_width=True)
            st.info("**Оранжевые точки** — неожиданно высокие продажи (промо, сезонный всплеск). "
                    "**Красные точки** — неожиданно низкие продажи (перебои, COVID).")

    st.divider()

    # ── Сводный анализ по всем рядам ─────────────────────────────────────────
    st.subheader("Сводный анализ по всем рядам")

    summary_tab, covid_tab, corr_tab = st.tabs([
        "Тренд и сезонность",
        "COVID-влияние",
        "Корреляции между рядами",
    ])

    # Сводная таблица
    with summary_tab:
        with st.spinner("Загружаем сводную аналитику..."):
            summary = fetch_analytics_summary()
        if summary and summary.get("summary"):
            df_sum = pd.DataFrame(summary["summary"])
            df_sum = df_sum.rename(columns={
                "category":                "Категория",
                "channel":                 "Канал",
                "n_obs":                   "Наблюд.",
                "mean":                    "Среднее, кг",
                "trend_direction":         "Тренд",
                "trend_significant":       "Тренд значим",
                "sens_slope_year":         "Наклон кг/год",
                "seasonality":             "Сезонность",
                "seasonality_significant": "Сез. значима",
                "eta_squared":             "Eta²",
            })
            df_sum["Тренд значим"]   = df_sum["Тренд значим"].map({True: "✔", False: "✘"})
            df_sum["Сез. значима"]   = df_sum["Сез. значима"].map({True: "✔", False: "✘"})
            st.dataframe(
                df_sum.style.map(
                    lambda v: "color: green" if v == "✔" else
                              "color: red"   if v == "✘" else "",
                    subset=["Тренд значим", "Сез. значима"],
                ),
                use_container_width=True, hide_index=True,
            )

    # COVID-влияние
    with covid_tab:
        channel_filter = st.selectbox(
            "Фильтр по каналу", ["Все каналы"] + sorted(channels), key="covid_ch"
        )
        ch_param = None if channel_filter == "Все каналы" else channel_filter
        with st.spinner("Считаем COVID-влияние..."):
            covid = fetch_covid_impact(ch_param)

        if covid:
            df_covid = pd.DataFrame(covid).rename(columns={
                "category":        "Категория",
                "channel":         "Канал",
                "covid_mean":      "Ср. в COVID, кг",
                "pre_covid_mean":  "Ср. 2019 (тот же пер.), кг",
                "base_mean":       "База (12 мес. до), кг",
                "drop_yoy_pct":    "Падение YoY, %",
                "drop_base_pct":   "Падение от базы, %",
                "recovery_months": "Восстановление, мес.",
                "impact_level":    "Уровень удара",
            })
            st.dataframe(df_covid, use_container_width=True, hide_index=True)

            # График падения YoY по категориям
            df_plot = pd.DataFrame(covid).dropna(subset=["drop_yoy_pct"])
            if not df_plot.empty:
                df_plot["label"] = df_plot["category"] + " | " + df_plot["channel"]
                df_plot = df_plot.sort_values("drop_yoy_pct")
                fig = go.Figure(go.Bar(
                    x=df_plot["drop_yoy_pct"],
                    y=df_plot["label"],
                    orientation="h",
                    marker_color=["#d62728" if v < 0 else "#2ca02c"
                                  for v in df_plot["drop_yoy_pct"]],
                    hovertemplate="%{y}<br>YoY: %{x:.1f}%<extra></extra>",
                ))
                fig.add_vline(x=0, line_color="black", line_width=1)
                fig.update_layout(
                    title="Изменение продаж в COVID vs 2019 (YoY), %",
                    xaxis_title="Изменение, %",
                    height=max(400, len(df_plot) * 22),
                    margin=dict(l=250),
                )
                st.plotly_chart(fig, use_container_width=True)
                st.info("**Зелёные** — категории которые выросли в COVID. "
                        "**Красные** — упали. Сортировка: снизу сильнейший удар.")

    # Корреляции
    with corr_tab:
        channel_filter_c = st.selectbox(
            "Фильтр по каналу", ["Все каналы"] + sorted(channels), key="corr_ch"
        )
        ch_param_c = None if channel_filter_c == "Все каналы" else channel_filter_c
        with st.spinner("Считаем корреляции..."):
            corr = fetch_correlation(ch_param_c)

        if corr and "error" not in corr:
            labels = corr["labels"]
            matrix = corr["matrix"]

            c1, c2 = st.columns(2)
            c1.metric("Рядов в анализе",      corr["n_series"])
            c2.metric("Средняя корреляция",    f"{corr['mean_correlation']:.3f}")

            # Тепловая карта
            fig = go.Figure(go.Heatmap(
                z=matrix, x=labels, y=labels,
                colorscale="RdYlGn", zmid=0, zmin=-1, zmax=1,
                text=[[f"{v:.2f}" for v in row] for row in matrix],
                texttemplate="%{text}",
                hovertemplate="%{y} × %{x}<br>r = %{z:.3f}<extra></extra>",
            ))
            fig.update_layout(
                title="Матрица корреляций между категориями",
                height=max(400, len(labels) * 40 + 100),
                xaxis=dict(tickangle=30),
            )
            st.plotly_chart(fig, use_container_width=True)

            col_top, col_bot = st.columns(2)
            with col_top:
                st.markdown("**Наиболее коррелированные пары:**")
                for p in corr.get("top_pairs", [])[:5]:
                    st.markdown(f"- {p['series_a']} × {p['series_b']}: **r={p['r']:.3f}**")
            with col_bot:
                st.markdown("**Наименее коррелированные:**")
                for p in corr.get("bottom_pairs", [])[:3]:
                    st.markdown(f"- {p['series_a']} × {p['series_b']}: **r={p['r']:.3f}**")

            st.info("**r > 0.8** — категории движутся синхронно (реагируют на одни факторы). "
                    "**r < 0.3** — категории независимы (хорошая диверсификация портфеля).")


# =============================================================================
# Страница: Прогноз
# =============================================================================


# =============================================================================
# Новые fetch-функции для ML v2
# =============================================================================

@st.cache_data(ttl=300)
def fetch_ml_v2_info(category: str, channel: str) -> dict:
    try:
        r = requests.get(f"{API_URL}/models/ml_v2/info/{category}/{channel}", timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception:
        return {}


@st.cache_data(ttl=300)
def fetch_model_comparison(category: str, channel: str) -> dict:
    try:
        r = requests.get(f"{API_URL}/models/comparison/{category}/{channel}", timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception:
        return {}


def fetch_forecast_ml_v2(
    category: str,
    channel: str,
    horizon: int,
    user_exog: dict | None = None,
    exog_file=None,
) -> dict:
    """Прогноз ML v2 — через файл или через JSON exog_data."""
    try:
        if exog_file is not None:
            r = requests.post(
                f"{API_URL}/forecast/upload",
                params={"category": category, "channel": channel,
                        "horizon": horizon, "model_class": "ml_v2"},
                files={"file": (exog_file.name, exog_file.getvalue(),
                                "application/octet-stream")},
                timeout=60,
            )
        else:
            r = requests.post(
                f"{API_URL}/forecast",
                json={"category": category, "channel": channel,
                      "horizon": horizon, "model_class": "ml_v2",
                      "exog_data": user_exog},
                timeout=60,
            )
        r.raise_for_status()
        return r.json()
    except requests.HTTPError as e:
        st.error(f"Ошибка API ({r.status_code}): {r.json().get('detail', str(e))}")
        return {}
    except Exception as e:
        st.error(f"Ошибка прогноза ML v2: {e}")
        return {}


# =============================================================================
# Вспомогательные функции страницы Прогноз
# =============================================================================

_ML_V2_EXOG_LABELS = {
    "Penetration":         "Пенетрация, %",
    "Spend per Trip":      "Трата за поход, руб.",
    "NT_CWD":              "CWD (взвешенная дистрибуция)",
    "NT_Price per kg":     "Цена за кг, руб.",
    "Volume per Trip":     "Объём за поход, кг",
    "NT_Avg Line":         "Среднее кол-во линеек",
    "Frequency":           "Частота покупок",
    "NT_Universe":         "Юниверс (числ.)",
    "MT_Universe percent": "Юниверс, %",
}


def _render_model_badge(comp: dict) -> None:
    """Баннер-рекомендация лучшей модели для ряда."""
    if not comp:
        return
    best  = comp.get("best_model", "")
    mape  = comp.get("best_mape")
    label_map = {
        "sarima":      "🏆 Рекомендуется: SARIMA / SARIMAX",
        "ml_v2":       "🏆 Рекомендуется: ML (XGBoost + экзогенные)",
    }
    label    = label_map.get(best, f"🏆 {best}")
    mape_str = f"  —  MAPE (тест, h=1..12) = **{mape:.1f}%**" if mape is not None else ""
    st.success(f"{label}{mape_str}")


def _render_comparison_table(available_pairs: list) -> None:
    """Таблица сравнения MAPE SARIMA vs ML для выбранных рядов."""
    rows = []
    for pair in available_pairs:
        cat, ch = pair["category"], pair["channel"]
        comp = fetch_model_comparison(cat, ch)
        if not comp:
            continue
        # API отдаёт ml_v2_no_exog и ml_v2_full
        ml_mape = comp.get("ml_v2_no_exog")
        ml_full = comp.get("ml_v2_full")
        rows.append({
            "Категория":      cat,
            "Канал":          ch,
            "SARIMA (тест)":  f"{comp['sarima_mape']:.1f}%" if comp.get("sarima_mape") else "—",
            "ML (тест)":      f"{ml_mape:.1f}%"            if ml_mape is not None       else "—",
            "ML c план. данными ↑": f"{ml_full:.1f}%"      if ml_full is not None        else "—",
            "✓ Лучшая": {
                "sarima": "SARIMA",
                "ml_v2":  "ML",
            }.get(comp.get("best_model", ""), "—"),
        })
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.caption(
            "MAPE на тестовом периоде (h=1..12 из одной точки T). "
            "«ML с план. данными ↑» — при вводе плановых значений переменных."
        )
    else:
        st.info("Данные сравнения моделей временно недоступны.")


# =============================================================================
# Страница: Прогноз
# =============================================================================

def page_forecast(series_list: list[dict]):
    st.title("Прогнозирование продаж")

    categories = sorted({s["category"] for s in series_list})
    channels   = sorted({s["channel"]  for s in series_list})

    # ── Параметры прогноза ───────────────────────────────────────────────────
    st.subheader("Параметры")
    col1, col2, col3 = st.columns([2, 2, 1])

    with col1:
        selected_cats = st.multiselect(
            "Категории", categories, default=[categories[0]],
            placeholder="Выберите категории...",
            key="fc_cats",
        )
    with col2:
        selected_chs = st.multiselect(
            "Каналы", channels, default=[channels[0]],
            placeholder="Выберите каналы...",
            key="fc_chs",
        )
    with col3:
        horizon = st.number_input(
            "Горизонт (мес.)", min_value=1, max_value=60, value=12, step=1
        )

    if not selected_cats or not selected_chs:
        st.info("Выберите хотя бы одну категорию и один канал.")
        return

    available_pairs = [
        s for s in series_list
        if s["category"] in selected_cats and s["channel"] in selected_chs
    ]
    if not available_pairs:
        st.warning("Нет данных для выбранной комбинации.")
        return

    st.caption(f"Выбрано рядов: **{len(available_pairs)}**")

    # ── Рекомендация лучшей модели (только для одного ряда) ─────────────────
    if len(available_pairs) == 1:
        cat0, ch0 = available_pairs[0]["category"], available_pairs[0]["channel"]
        comp0 = fetch_model_comparison(cat0, ch0)
        _render_model_badge(comp0)

    # ── Выбор класса модели ──────────────────────────────────────────────────
    st.subheader("Модель")
    model_class = st.radio(
        "Выберите класс модели",
        options=["econometric", "ml_v2"],
        format_func=lambda x: {
            "econometric": "📈 Эконометрика (SARIMA / SARIMAX)",
            "ml_v2":       "🤖 ML (XGBoost + экзогенные переменные)",
        }[x],
        horizontal=True,
        key="model_class_radio",
    )

    # ── Блок экзогенных ──────────────────────────────────────────────────────
    exog_file    = None
    user_exog_kv = None

    if model_class == "econometric":
        with st.expander("📎 Загрузить экзогенные переменные (необязательно)"):
            st.markdown("""
            Если у вас есть плановые значения внешних факторов на период прогноза,
            загрузите их в формате Excel или CSV.

            **Требования к файлу:**
            - Каждая переменная отдельная колонка
            - Количество строк = горизонт прогноза
            - Допустимые названия колонок ниже:

            | Колонка | Описание |
            |---|---|
            | `NT_Avg Line` | Среднее кол-во линеек |
            | `NT_Price per kg` | Цена за кг |
            | `NT_Universe` | Юниверс (численный) |
            | `MT_Universe percent` | Юниверс (%) |
            | `Frequency` | Частота покупок |
            | `Penetration` | Пенетрация |
            | `Spend per Trip` | Трата за поход |
            | `Volume per Trip` | Объём за поход |
            | `Total Mixed Chains - VoD (Vol)` | VoD объём |
            | `NT_CWD` | CWD |

            Передавать все колонки необязательно — только те что есть.
            Недостающие переменные будут спрогнозированы автоматически.
            """)
            exog_file = st.file_uploader(
                "Загрузить файл", type=["xlsx", "xls", "csv"],
                key="exog_upload_eco",
            )
            if exog_file:
                try:
                    preview = pd.read_excel(exog_file) if exog_file.name.endswith(
                        (".xlsx", ".xls")) else pd.read_csv(exog_file)
                    st.success(f"Файл загружен: {len(preview)} строк, "
                               f"{len(preview.columns)} колонок")
                    st.dataframe(preview.head(3), use_container_width=True)
                    exog_file.seek(0)
                except Exception as e:
                    st.error(f"Ошибка чтения файла: {e}")
                    exog_file = None

    else:  # ml_v2
        # Загружаем список переменных из API
        ml_info = {}
        if len(available_pairs) == 1:
            ml_info = fetch_ml_v2_info(
                available_pairs[0]["category"], available_pairs[0]["channel"]
            )
        selected_exog_vars = ml_info.get("selected_exog", [])

        with st.expander("📎 Плановые значения переменных (необязательно)", expanded=False):
            st.markdown(
                "Если вы знаете плановые значения переменных — введите их ниже. "
                "Это улучшит точность прогноза. "
                "Незаполненные переменные модель заполнит автоматически (режим **no_exog**)."
            )

            if not selected_exog_vars:
                st.info("Список переменных модели временно недоступен.")
                exog_file = st.file_uploader(
                    "Загрузить файл с плановыми значениями",
                    type=["xlsx", "xls", "csv"],
                    key="exog_upload_ml_fallback",
                )
            else:
                input_mode = st.radio(
                    "Способ ввода",
                    ["Ввести вручную", "Загрузить файл"],
                    horizontal=True,
                    key="exog_input_mode",
                )

                if input_mode == "Загрузить файл":
                    exog_file = st.file_uploader(
                        "Файл Excel или CSV",
                        type=["xlsx", "xls", "csv"],
                        key="exog_upload_ml",
                    )
                    if exog_file:
                        try:
                            preview = pd.read_excel(exog_file) if exog_file.name.endswith(
                                (".xlsx", ".xls")) else pd.read_csv(exog_file)
                            st.success(f"Файл загружен: {len(preview)} строк")
                            st.dataframe(preview.head(3), use_container_width=True)
                            exog_file.seek(0)
                        except Exception as e:
                            st.error(f"Ошибка чтения файла: {e}")
                            exog_file = None
                else:
                    st.caption(
                        f"Переменные модели ({len(selected_exog_vars)} шт., "
                        f"порядок по CCF-корреляции):"
                    )
                    user_exog_kv = {}
                    for col in selected_exog_vars:
                        label = _ML_V2_EXOG_LABELS.get(col, col)
                        raw = st.text_input(
                            label,
                            placeholder=f"Через запятую, {horizon} значений. Пример: 100, 102, 103",
                            key=f"exog_input_{col}",
                            help=(
                                f"Плановые значения {col} на {horizon} месяцев. "
                                "Оставьте пустым — модель использует последнее известное значение."
                            ),
                        )
                        if raw.strip():
                            try:
                                vals = [float(v.strip()) for v in raw.split(",") if v.strip()]
                                if vals:
                                    user_exog_kv[col] = vals
                            except ValueError:
                                st.warning(f"Некорректные значения для «{label}»")

                    if user_exog_kv:
                        st.caption(
                            f"✓ Заполнено: {len(user_exog_kv)} из {len(selected_exog_vars)} переменных"
                        )
                    else:
                        st.caption("Прогноз будет в режиме no_exog (без плановых данных)")

    # ── Информация о моделях ─────────────────────────────────────────────────
    with st.expander("ℹ️ Сравнение моделей (MAPE на тестовом периоде)"):
        _render_comparison_table(available_pairs)

    # ── Кнопка запуска ───────────────────────────────────────────────────────
    if st.button("Построить прогноз", type="primary", use_container_width=True):
        forecasts = []
        histories = []
        errors    = []

        progress = st.progress(0, text="Строим прогнозы...")
        for idx, pair in enumerate(available_pairs):
            cat, ch = pair["category"], pair["channel"]

            h = fetch_history(cat, ch)
            if h:
                histories.append(h)

            if model_class == "ml_v2":
                fc = fetch_forecast_ml_v2(
                    cat, ch, horizon,
                    user_exog=user_exog_kv if user_exog_kv else None,
                    exog_file=exog_file,
                )
            else:
                fc = fetch_forecast(cat, ch, horizon, exog_file)

            if fc:
                forecasts.append(fc)
            else:
                errors.append(f"{cat} | {ch}")

            if exog_file:
                exog_file.seek(0)

            progress.progress(
                (idx + 1) / len(available_pairs),
                text=f"Прогноз: {cat} | {ch}",
            )
        progress.empty()

        if errors:
            st.warning(f"Не удалось построить прогноз для: {', '.join(errors)}")

        if not forecasts:
            st.error("Ни одного прогноза не получено.")
            return

        # ── Результаты ───────────────────────────────────────────────────────
        st.divider()

        if len(forecasts) == 1:
            fc = forecasts[0]
            h  = histories[0] if histories else {}

            c1, c2, c3 = st.columns(3)
            c1.metric("Модель",   fc.get("model_type", "—"))
            c2.metric("Горизонт", f"{horizon} мес.")
            c3.metric(
                "MAPE (тест, h=1..12)",
                f"{fc['test_mape']:.1f}%" if fc.get("test_mape") else "—",
                help="MAPE на тестовом периоде: прогноз h=1..12 из одной точки T",
            )

            if fc.get("exog_cols"):
                st.caption(f"Переданы плановые данные: {', '.join(fc['exog_cols'])}")
            elif model_class == "ml_v2":
                st.caption("Режим без плановых данных")

            st.plotly_chart(
                build_forecast_chart(h, fc),
                use_container_width=True,
            )

            with st.expander("Таблица прогноза"):
                fc_df = pd.DataFrame(fc["forecast"])
                fc_df.columns = ["Дата", "Прогноз, кг", "Нижн. 80% ДИ", "Верхн. 80% ДИ"]
                fc_df["Прогноз, кг"]   = fc_df["Прогноз, кг"].map("{:,.0f}".format)
                fc_df["Нижн. 80% ДИ"] = fc_df["Нижн. 80% ДИ"].map("{:,.0f}".format)
                fc_df["Верхн. 80% ДИ"]= fc_df["Верхн. 80% ДИ"].map("{:,.0f}".format)
                st.dataframe(fc_df, use_container_width=True, hide_index=True)

                csv = pd.DataFrame(fc["forecast"]).to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Скачать прогноз (CSV)", csv,
                    file_name=f"forecast_{fc['category']}_{fc['channel']}.csv",
                    mime="text/csv",
                )

        else:
            tabs_labels = (
                [f"{fc['category']} | {fc['channel']}" for fc in forecasts]
                + ["Суммарно"]
            )
            tabs = st.tabs(tabs_labels)

            for tab, fc, h in zip(tabs[:-1], forecasts, histories):
                with tab:
                    col1, col2 = st.columns(2)
                    col1.metric("Модель", fc.get("model_type", "—"))
                    col2.metric(
                        "MAPE (тест, h=1..12)",
                        f"{fc['test_mape']:.1f}%" if fc.get("test_mape") else "—",
                    )
                    st.plotly_chart(
                        build_forecast_chart(h, fc),
                        use_container_width=True,
                    )
                    with st.expander("Таблица"):
                        fc_df = pd.DataFrame(fc["forecast"])
                        st.dataframe(fc_df, use_container_width=True, hide_index=True)
                        csv = fc_df.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            "CSV", csv,
                            file_name=f"forecast_{fc['category']}_{fc['channel']}.csv",
                            mime="text/csv",
                        )

            with tabs[-1]:
                agg_label = f"{', '.join(selected_cats)} | {', '.join(selected_chs)}"
                st.plotly_chart(
                    build_aggregate_forecast_chart(histories, forecasts, agg_label),
                    use_container_width=True,
                )
                st.caption(
                    "Суммарный прогноз = сумма прогнозов по выбранным рядам. "
                    "Доверительный интервал = сумма границ каждого ряда."
                )
                with st.expander("Суммарная таблица"):
                    fc_df = pd.concat([pd.DataFrame(fc["forecast"]) for fc in forecasts])
                    agg_df = fc_df.groupby("date").agg(
                        forecast=("forecast", "sum"),
                        lower_80=("lower_80", "sum"),
                        upper_80=("upper_80", "sum"),
                    ).reset_index()
                    agg_df.columns = ["Дата", "Прогноз, кг", "Нижн. 80% ДИ", "Верхн. 80% ДИ"]
                    st.dataframe(agg_df, use_container_width=True, hide_index=True)
                    csv = agg_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "Скачать суммарный прогноз (CSV)", csv,
                        file_name="forecast_aggregate.csv",
                        mime="text/csv",
                    )


# =============================================================================
# Навигация
# =============================================================================

def main():
    st.sidebar.title("Mars Forecasting")
    st.sidebar.divider()

    page = st.sidebar.radio(
        "Раздел",
        ["Аналитика", "Прогноз"],
        label_visibility="collapsed",
    )

    st.sidebar.divider()
    st.sidebar.caption("API: " + API_URL)

    try:
        r = requests.get(f"{API_URL}/health", timeout=3)
        if r.status_code == 200:
            st.sidebar.success("API доступен")
        else:
            st.sidebar.error("API недоступен")
    except Exception:
        st.sidebar.error("API недоступен")
        st.error("Не удалось подключиться к API.")
        return

    series_list = fetch_series()
    if not series_list:
        st.error("Нет данных. Проверь подключение к API.")
        return

    st.sidebar.caption(f"Рядов в базе: {len(series_list)}")

    if page == "Аналитика":
        page_analytics(series_list)
    elif page == "Прогноз":
        page_forecast(series_list)


if __name__ == "__main__":
    main()