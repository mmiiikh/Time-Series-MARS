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

st.set_page_config(
    page_title="Mars Sales Forecasting",
    layout="wide",
)



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
            params={"diff": diff}, timeout=15
        )
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
    """График исторических данных для одного или нескольких рядов."""
    fig = go.Figure()
    for h in histories:
        label = f"{h['category']} | {h['channel']}"
        fig.add_trace(go.Scatter(
            x=h["dates"], y=h["values"],
            mode="lines", name=label,
            hovertemplate="%{x}<br>%{y:,.0f} кг<extra>" + label + "</extra>",
        ))
    fig.update_layout(
        title=title or "Исторические данные",
        xaxis_title="Дата", yaxis_title="Объём продаж, кг",
        hovermode="x unified", legend=dict(orientation="h", y=-0.2),
        height=420,
    )
    return fig


def build_stl_chart(stl: dict) -> go.Figure:
    """4-компонентный STL-график."""
    fig = go.Figure()
    components = [
        ("observed",  "Факт",        "#1f77b4"),
        ("trend",     "Тренд",       "#d62728"),
        ("seasonal",  "Сезонность",  "#ff7f0e"),
        ("resid",     "Остаток",     "#9467bd"),
    ]
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



def page_analytics(series_list: list[dict]):
    st.title("Аналитика временных рядов")

    categories = sorted({s["category"] for s in series_list})
    channels   = sorted({s["channel"]  for s in series_list})

    # Фильтры
    col1, col2 = st.columns(2)
    with col1:
        selected_cats = st.multiselect(
            "Категории", categories, default=[categories[0]],
            placeholder="Выберите категории..."
        )
    with col2:
        selected_chs = st.multiselect(
            "Каналы", channels, default=[channels[0]],
            placeholder="Выберите каналы..."
        )

    if not selected_cats or not selected_chs:
        st.info("Выберите хотя бы одну категорию и один канал.")
        return

    # Доступные пары для выбранных фильтров
    available_pairs = [
        s for s in series_list
        if s["category"] in selected_cats and s["channel"] in selected_chs
    ]
    if not available_pairs:
        st.warning("Нет данных для выбранной комбинации.")
        return

    # ── Исторические графики ─────────────────────────────────────────────────
    st.subheader("Исторические данные")
    histories = []
    for pair in available_pairs:
        h = fetch_history(pair["category"], pair["channel"])
        if h:
            histories.append(h)

    if histories:
        st.plotly_chart(
            build_history_chart(histories, "Объём продаж по выбранным рядам"),
            use_container_width=True,
        )

    # ── Базовые статистики ───────────────────────────────────────────────────
    with st.expander("📋 Базовые статистики"):
        stats_df = pd.DataFrame([{
            "Категория": s["category"],
            "Канал":     s["channel"],
            "Наблюдений": s["n_obs"],
            "С":         s["date_from"],
            "По":        s["date_to"],
        } for s in available_pairs])
        st.dataframe(stats_df, use_container_width=True, hide_index=True)

    # ── Аналитика одного ряда ────────────────────────────────────────────────
    st.subheader("Детальный анализ ряда")
    st.caption("Выберите один ряд для детального анализа")

    pair_labels = [f"{s['category']} | {s['channel']}" for s in available_pairs]
    selected_label = st.selectbox("Ряд для анализа", pair_labels)
    selected_pair  = available_pairs[pair_labels.index(selected_label)]
    cat, ch = selected_pair["category"], selected_pair["channel"]

    tab_stl, tab_stat, tab_acf = st.tabs(
        ["STL-декомпозиция", "Стационарность", "ACF / PACF"]
    )

    # STL
    with tab_stl:
        with st.spinner("Считаем STL-декомпозицию..."):
            stl = fetch_stl(cat, ch)
        if stl and "error" not in stl:
            m = stl["metrics"]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Сила сезонности Fs", f"{m['Fs_seasonal']:.3f}")
            c2.metric("Сила тренда Ft",     f"{m['Ft_trend']:.3f}")
            c3.metric("R²",                 f"{m['R2']:.3f}")
            c4.metric("Амплитуда сезонности", f"{m['seas_amplitude']:,.0f} кг")
            st.plotly_chart(build_stl_chart(stl), use_container_width=True)
        elif stl and "error" in stl:
            st.warning(stl["error"])

    # Стационарность
    with tab_stat:
        with st.spinner("Запускаем тесты стационарности..."):
            stat = fetch_stationarity(cat, ch)
        if stat and "tests" in stat:
            labels = {
                "original": "Исходный ряд",
                "d=1":      "Первая разность (d=1)",
                "D=1":      "Сезонная разность (D=1)",
                "d=1+D=1":  "Обе разности (d=1, D=1)",
            }
            rows = []
            for diff_key, result in stat["tests"].items():
                rows.append({
                    "Уровень":         labels.get(diff_key, diff_key),
                    "ADF stat":        result["adf_stat"],
                    "ADF p-value":     result["adf_p"],
                    "KPSS stat":       result["kpss_stat"],
                    "KPSS p-value":    result["kpss_p"],
                    "Вывод":           result["conclusion"],
                    "Стационарен":     "✔" if result["is_stationary"] else "✘",
                })
            df_stat = pd.DataFrame(rows)
            st.dataframe(
                df_stat.style.applymap(
                    lambda v: "color: green" if v == "✔" else "color: red" if v == "✘" else "",
                    subset=["Стационарен"]
                ),
                use_container_width=True, hide_index=True,
            )

    # ACF / PACF
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
        elif acf_data and "error" in acf_data:
            st.warning(acf_data["error"])


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

    # ── Экзогенные переменные ────────────────────────────────────────────────
    with st.expander("📎 Загрузить экзогенные переменные (необязательно)"):
        st.markdown("""
        Если у вас есть плановые значения внешних факторов на период прогноза,
        загрузите их в формате Excel или CSV.

        **Требования к файлу:**
        - Каждая переменная — отдельная колонка
        - Количество строк = горизонт прогноза
        - Допустимые названия колонок:

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
            key="exog_upload",
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

    if exog_file is None:
        exog_file = None

    # ── Информация об обученных моделях ─────────────────────────────────────
    with st.expander("Информация о моделях"):
        model_rows = []
        for pair in available_pairs:
            info = fetch_model_info(pair["category"], pair["channel"])
            if info:
                model_rows.append({
                    "Категория":     pair["category"],
                    "Канал":         pair["channel"],
                    "Модель":        info.get("best_model", "—"),
                    "Порядок":       str(info.get("order", "—")),
                    "Сез. порядок":  str(info.get("seasonal_order", "—")),
                    "Экзогенные":    ", ".join(info.get("exog_cols", [])) or "—",
                    "MAPE (тест)":   f"{info['mape']:.1f}%" if info.get("mape") else "—",
                })
        if model_rows:
            st.dataframe(pd.DataFrame(model_rows),
                         use_container_width=True, hide_index=True)

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

            fc = fetch_forecast(cat, ch, horizon, exog_file)
            if fc:
                forecasts.append(fc)
            else:
                errors.append(f"{cat} | {ch}")

            if exog_file:
                exog_file.seek(0)

            progress.progress(
                (idx + 1) / len(available_pairs),
                text=f"Прогноз: {cat} | {ch}"
            )
        progress.empty()

        if errors:
            st.warning(f"Не удалось построить прогноз для: {', '.join(errors)}")

        if not forecasts:
            st.error("Ни одного прогноза не получено.")
            return

        # ── Результаты ───────────────────────────────────────────────────────
        st.divider()

        # Если выбран один ряд — показываем его отдельно
        if len(forecasts) == 1:
            fc = forecasts[0]
            h  = histories[0] if histories else {}

            # Метрики
            c1, c2, c3 = st.columns(3)
            c1.metric("Модель",    fc.get("model_type", "—"))
            c2.metric("Горизонт",  f"{horizon} мес.")
            c3.metric("MAPE (обучение)",
                      f"{fc['train_mape']:.1f}%" if fc.get("train_mape") else "—")

            if fc.get("exog_cols"):
                st.caption(f"Экзогенные переменные: {', '.join(fc['exog_cols'])}")

            st.plotly_chart(
                build_forecast_chart(h, fc),
                use_container_width=True,
            )

            # Таблица прогноза
            with st.expander("Таблица прогноза"):
                fc_df = pd.DataFrame(fc["forecast"])
                fc_df.columns = ["Дата", "Прогноз, кг", "Нижн. 80% ДИ", "Верхн. 80% ДИ"]
                fc_df["Прогноз, кг"]    = fc_df["Прогноз, кг"].map("{:,.0f}".format)
                fc_df["Нижн. 80% ДИ"]  = fc_df["Нижн. 80% ДИ"].map("{:,.0f}".format)
                fc_df["Верхн. 80% ДИ"] = fc_df["Верхн. 80% ДИ"].map("{:,.0f}".format)
                st.dataframe(fc_df, use_container_width=True, hide_index=True)

                # Скачать CSV
                csv = pd.DataFrame(fc["forecast"]).to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇ Скачать прогноз (CSV)", csv,
                    file_name=f"forecast_{fc['category']}_{fc['channel']}.csv",
                    mime="text/csv",
                )

        # Если выбрано несколько рядов — показываем каждый + суммарный
        else:
            tabs_labels = (
                [f"{fc['category']} | {fc['channel']}" for fc in forecasts]
                + ["∑ Суммарно"]
            )
            tabs = st.tabs(tabs_labels)

            # Индивидуальные ряды
            for tab, fc, h in zip(tabs[:-1], forecasts, histories):
                with tab:
                    col1, col2 = st.columns(2)
                    col1.metric("Модель",   fc.get("model_type", "—"))
                    col2.metric("MAPE",
                                f"{fc['train_mape']:.1f}%" if fc.get("train_mape") else "—")
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

            # Суммарный прогноз
            with tabs[-1]:
                agg_label = (
                    f"{', '.join(selected_cats)} | {', '.join(selected_chs)}"
                )
                st.plotly_chart(
                    build_aggregate_forecast_chart(histories, forecasts, agg_label),
                    use_container_width=True,
                )
                st.caption(
                    "Суммарный прогноз = сумма прогнозов по выбранным рядам. "
                    "Доверительный интервал = сумма границ каждого ряда."
                )

                # Суммарная таблица
                with st.expander("📋 Суммарная таблица"):
                    fc_df = pd.concat([pd.DataFrame(fc["forecast"]) for fc in forecasts])
                    agg_df = fc_df.groupby("date").agg(
                        forecast=("forecast", "sum"),
                        lower_80=("lower_80", "sum"),
                        upper_80=("upper_80", "sum"),
                    ).reset_index()
                    agg_df.columns = [
                        "Дата", "Прогноз, кг", "Нижн. 80% ДИ", "Верхн. 80% ДИ"
                    ]
                    st.dataframe(agg_df, use_container_width=True, hide_index=True)
                    csv = agg_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "⬇ Скачать суммарный прогноз (CSV)", csv,
                        file_name="forecast_aggregate.csv",
                        mime="text/csv",
                    )


# =============================================================================
# Навигация
# =============================================================================

def main():
    st.sidebar.title("🍫 Mars Forecasting")
    st.sidebar.divider()

    page = st.sidebar.radio(
        "Раздел",
        ["Аналитика", "Прогноз"],
        label_visibility="collapsed",
    )

    st.sidebar.divider()
    st.sidebar.caption("API: " + API_URL)

    # Проверяем доступность API
    try:
        r = requests.get(f"{API_URL}/health", timeout=3)
        if r.status_code == 200:
            st.sidebar.success("API доступен ✔")
        else:
            st.sidebar.error("API недоступен ✘")
    except Exception:
        st.sidebar.error("API недоступен ✘")
        st.error("Не удалось подключиться к API. Убедись, что FastAPI запущен.")
        return

    # Загружаем список рядов
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