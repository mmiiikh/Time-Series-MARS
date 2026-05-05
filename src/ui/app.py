import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from io import BytesIO

import os as _os
try:
    API_URL = st.secrets.get("API_URL",
              _os.getenv("API_URL", "http://localhost:8000"))
except Exception:
    API_URL = _os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Mars Sales Forecasting", layout="wide")


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
        r = requests.get(f"{API_URL}/series/{category}/{channel}/acf_pacf",params={"diff": diff}, timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"Ошибка ACF/PACF: {e}")
        return {}


@st.cache_data(ttl=300)
def fetch_model_comparison(category: str, channel: str) -> dict:
    try:
        r = requests.get(f"{API_URL}/models/comparison/{category}/{channel}", timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception:
        return {}


@st.cache_data(ttl=300)
def fetch_ml_v2_info(category: str, channel: str) -> dict:
    try:
        r = requests.get(f"{API_URL}/models/ml_v2/info/{category}/{channel}", timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception:
        return {}


@st.cache_data(ttl=60)
def fetch_hp_filter(category: str, channel: str) -> dict:
    try:
        r = requests.get(f"{API_URL}/series/{category}/{channel}/hp_filter", timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception:
        return {}


@st.cache_data(ttl=60)
def fetch_fft(category: str, channel: str) -> dict:
    try:
        r = requests.get(f"{API_URL}/series/{category}/{channel}/fft", timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception:
        return {}


@st.cache_data(ttl=60)
def fetch_mann_kendall(category: str, channel: str) -> dict:
    try:
        r = requests.get(f"{API_URL}/series/{category}/{channel}/mann_kendall", timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception:
        return {}


@st.cache_data(ttl=60)
def fetch_structural_breaks(category: str, channel: str) -> dict:
    try:
        r = requests.get(f"{API_URL}/series/{category}/{channel}/structural_breaks", timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception:
        return {}


@st.cache_data(ttl=60)
def fetch_seasonal_subseries(category: str, channel: str) -> dict:
    try:
        r = requests.get(f"{API_URL}/series/{category}/{channel}/seasonal_subseries", timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception:
        return {}


@st.cache_data(ttl=60)
def fetch_seasonality_test(category: str, channel: str) -> dict:
    try:
        r = requests.get(f"{API_URL}/series/{category}/{channel}/seasonality_test", timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception:
        return {}


@st.cache_data(ttl=60)
def fetch_anomalies(category: str, channel: str, method: str = "stl") -> dict:
    try:
        r = requests.get(f"{API_URL}/series/{category}/{channel}/anomalies",params={"method": method}, timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception:
        return {}


@st.cache_data(ttl=60)
def fetch_covid_impact(channel: str = None) -> list:
    try:
        params = {"channel": channel} if channel else {}
        r = requests.get(f"{API_URL}/analytics/covid_impact", params=params, timeout=20)
        r.raise_for_status()
        return r.json()
    except Exception:
        return []


@st.cache_data(ttl=60)
def fetch_correlation(channel: str = None) -> dict:
    try:
        params = {"channel": channel} if channel else {}
        r = requests.get(f"{API_URL}/analytics/correlation", params=params, timeout=20)
        r.raise_for_status()
        return r.json()
    except Exception:
        return {}


@st.cache_data(ttl=60)
def fetch_analytics_summary() -> dict:
    try:
        r = requests.get(f"{API_URL}/analytics/summary", timeout=30)
        r.raise_for_status()
        return r.json()
    except Exception:
        return {}


def fetch_forecast(category:str, channel:str, horizon:int,
                   model_class:str = "econometric",
                   user_exog:dict = None,
                   exog_file=None) -> dict:
    try:
        if exog_file is not None:
            r = requests.post(
                f"{API_URL}/forecast/upload",
                params={"category": category, "channel": channel,
                        "horizon": horizon, "model_class": model_class},
                files={"file": (exog_file.name, exog_file.getvalue(),
                                "application/octet-stream")},
                timeout=60)
        else:
            _timeout = 180 if model_class == "ml_v2" else 60
            r = requests.post(
                f"{API_URL}/forecast",
                json={"category": category, "channel": channel,
                      "horizon": horizon, "model_class": model_class,
                      "exog_data": user_exog},
                timeout=_timeout)
        r.raise_for_status()
        return r.json()
    except requests.HTTPError as e:
        try:
            detail = r.json().get("detail", str(e))
        except Exception:
            detail = str(e)
        st.error(f"Ошибка API ({r.status_code}): {detail}")
        return {}
    except Exception as e:
        st.error(f"Ошибка прогноза: {e}")
        return {}


@st.cache_data(ttl=300)
def fetch_lstm_info(variant: str = None) -> dict:
    try:
        params = {"variant": variant} if variant else {}
        r = requests.get(f"{API_URL}/models/lstm/info", params=params, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception:
        return {}


@st.cache_data(ttl=300)
def fetch_lstm_variants() -> list[str]:
    try:
        r = requests.get(f"{API_URL}/models/lstm/variants", timeout=5)
        r.raise_for_status()
        return r.json().get("available_variants", [])
    except Exception:
        return []


@st.cache_data(ttl=300)
def fetch_full_comparison(category: str, channel: str) -> dict:
    try:
        r = requests.get(f"{API_URL}/models/full_comparison/{category}/{channel}", timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception:
        return {}


def fetch_forecast_lstm(category:str, channel:str, horizon:int,variant:str = None, user_exog: dict = None) -> dict:
    try:
        payload = {
            "category":category,
            "channel":channel,
            "horizon":horizon,
            "model_class": "lstm",
            "exog_data":user_exog}
        r = requests.post(f"{API_URL}/forecast", json=payload, timeout=120)
        r.raise_for_status()
        return r.json()
    except requests.HTTPError as e:
        try:
            detail = r.json().get("detail", str(e))
        except Exception:
            detail = str(e)
        st.error(f"Ошибка LSTM прогноза: {detail}")
        return {}
    except Exception as e:
        st.error(f"Ошибка LSTM прогноза: {e}")
        return {}



def fetch_aggregate_forecast(pairs: list[dict], model_class: str,horizon: int, label: str = "") -> dict:
    try:
        payload = {
            "pairs": pairs,
            "model_class": model_class,
            "horizon":horizon,
            "label":label or f"Агрегация ({len(pairs)} рядов)"}
        r = requests.post(f"{API_URL}/forecast/aggregate", json=payload, timeout=180)
        r.raise_for_status()
        return r.json()
    except requests.HTTPError as e:
        try:
            detail = r.json().get("detail", str(e))
        except Exception:
            detail = str(e)
        st.error(f"Ошибка агрегированного прогноза: {detail}")
        return {}
    except Exception as e:
        st.error(f"Ошибка агрегированного прогноза: {e}")
        return {}


def build_history_chart(histories: list[dict], title: str = "") -> go.Figure:
    fig = go.Figure()
    for h in histories:
        label = f"{h['category']} | {h['channel']}"
        fig.add_trace(go.Scatter(
            x=h["dates"], y=h["values"], mode="lines", name=label,
            hovertemplate="%{x}<br>%{y:,.0f} кг<extra>" + label + "</extra>"))
    fig.update_layout(
        title=title or "Исторические данные",
        xaxis_title="Дата", yaxis_title="Объём продаж, кг",
        hovermode="x unified", legend=dict(orientation="h", y=-0.2), height=420)
    return fig


def build_stl_chart(stl: dict) -> go.Figure:
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                        subplot_titles=["Факт", "Тренд", "Сезонность", "Остаток"],
                        vertical_spacing=0.06)
    components = [
        ("observed","Факт","#1f77b4"),
        ("trend","Тренд","#d62728"),
        ("seasonal","Сезонность","#ff7f0e"),
        ("resid","Остаток", "#9467bd")]
    for row, (key, label, color) in enumerate(components, 1):
        fig.add_trace(go.Scatter(x=stl["dates"], y=stl[key], mode="lines", name=label,
            line=dict(color=color, width=1.5),
            hovertemplate="%{x}<br>%{y:,.0f}<extra></extra>"), row=row, col=1)
    fig.update_layout(height=700, showlegend=False,title=f"STL-декомпозиция: {stl['category']} | {stl['channel']}")
    return fig


def build_acf_pacf_chart(data: dict) -> go.Figure:
    fig = make_subplots(rows=1, cols=2, subplot_titles=["ACF", "PACF"])
    cb  = data["conf_bound"]
    for col, (vals, name) in enumerate([(data["acf"], "ACF"), (data["pacf"], "PACF")], 1):
        lags = data["lags"]
        for lag, val in zip(lags, vals):
            fig.add_trace(go.Scatter(
                x=[lag, lag], y=[0, val], mode="lines",
                line=dict(color="#1f77b4", width=1.5), showlegend=False,
                hovertemplate=f"Lag {lag}: {val:.3f}<extra></extra>"), row=1, col=col)
        fig.add_hline(y=cb,line_dash="dash", line_color="red",annotation_text="95% ДИ", row=1, col=col)
        fig.add_hline(y=-cb,line_dash="dash", line_color="red", row=1, col=col)
        fig.add_hline(y=0,line_color="black", line_width=0.8, row=1, col=col)
        for sl in data.get("seasonal_lags", []):
            fig.add_vline(x=sl, line_dash="dot", line_color="grey", opacity=0.6, row=1, col=col)
    fig.update_layout(height=380, showlegend=False, title=f"ACF/PACF: {data['diff']}")
    return fig


def build_forecast_chart(history: dict, forecast: dict,
                          title: str = "") -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=history.get("dates", []), y=history.get("values", []),
        mode="lines", name="История", line=dict(color="#1f77b4", width=2),
        hovertemplate="%{x}<br>%{y:,.0f} кг<extra>История</extra>"))
    fc_points = forecast.get("forecast", [])
    if fc_points:
        dates = [p["date"] for p in fc_points]
        values = [p["forecast"] for p in fc_points]
        lower = [p["lower_80"] for p in fc_points]
        upper = [p["upper_80"] for p in fc_points]
        fig.add_trace(go.Scatter(
            x=dates+dates[::-1], y=upper+lower[::-1],
            fill="toself", fillcolor="rgba(214,39,40,0.15)",
            line=dict(color="rgba(255,255,255,0)"), name="80% ДИ", hoverinfo="skip"))
        fig.add_trace(go.Scatter(
            x=dates, y=values, mode="lines+markers", name="Прогноз",
            line=dict(color="#d62728", width=2, dash="dash"), marker=dict(size=5),
            hovertemplate="%{x}<br>%{y:,.0f} кг<extra>Прогноз</extra>"))
        if history.get("dates"):
            fig.add_vline(x=history["dates"][-1], line_dash="dot",line_color="grey", opacity=0.7)
    fig.update_layout(
        title=title or f"Прогноз:{forecast.get('category','')} | {forecast.get('channel','')}",
        xaxis_title="Дата", yaxis_title="Объём продаж кг",
        hovermode="x unified", legend=dict(orientation="h", y=-0.2), height=450)
    return fig


def build_aggregate_chart(agg_result: dict, history_sum: dict = None) -> go.Figure:
    fig = go.Figure()
    if history_sum and history_sum.get("dates"):
        fig.add_trace(go.Scatter(
            x=history_sum["dates"], y=history_sum["values"],
            mode="lines", name="История (сумма)", line=dict(color="#1f77b4", width=2),
            hovertemplate="%{x}<br>%{y:,.0f} кг<extra>История</extra>"))

    fc_points = agg_result.get("forecast", [])
    if fc_points:
        dates = [p["date"] for p in fc_points]
        values = [p["forecast"] for p in fc_points]
        lower = [p["lower_80"] for p in fc_points]
        upper = [p["upper_80"] for p in fc_points]
        fig.add_trace(go.Scatter(
            x=dates + dates[::-1], y=upper + lower[::-1],
            fill="toself", fillcolor="rgba(214,39,40,0.15)",
            line=dict(color="rgba(255,255,255,0)"), name="80% ДИ", hoverinfo="skip"))
        fig.add_trace(go.Scatter(
            x=dates, y=values, mode="lines+markers",
            name="Прогноз (сумма)", line=dict(color="#d62728", width=2, dash="dash"),
            marker=dict(size=5), hovertemplate="%{x}<br>%{y:,.0f} кг<extra>Прогноз</extra>"))
        if history_sum and history_sum.get("dates"):
            fig.add_vline(x=history_sum["dates"][-1], line_dash="dot",
                          line_color="grey", opacity=0.7)

    fig.update_layout(
        title=agg_result.get("label", "Агрегированный прогноз"),
        xaxis_title="Дата", yaxis_title="Объём продаж кг",
        hovermode="x unified", legend=dict(orientation="h", y=-0.2), height=460)
    return fig



_ML_V2_EXOG_LABELS = {
    "Penetration": "Пенетрация, %",
    "Spend per Trip": "Трата за поход руб.",
    "NT_CWD": "CWD (взв. дистрибуция)",
    "NT_Price per kg": "Цена за кг руб.",
    "Volume per Trip": "Объём за поход кг",
    "NT_Avg Line": "Среднее кол-во линеек",
    "Frequency": "Частота покупок",
    "NT_Universe": "Число магазинов",
    "MT_Universe percent": "Доля магазинов где представлена категория"}

_ECONOMETRIC_LABELS = {
    "Naive": "Наивный сезонный прогноз",
    "ARIMA": "ARIMA",
    "SARIMA": "SARIMA",
    "SARIMAX":"SARIMAX (с экзог)",
    "Prophet": "Prophet (годовая сезонность)"}


def _render_model_badge(comp: dict, full_comp: dict = None) -> None:
    if full_comp:
        best = full_comp.get("best_overall") or full_comp.get("best_model")
        mape = full_comp.get("best_mape")
        mape_s = f", MAPE = **{mape:.1f}%**" if mape else ""
        if best in ("lstm", "LSTM"):
            variant = full_comp.get("lstm_variant", "exog")
            st.success(f"Рекомендуется: LSTM ({variant}){mape_s}")
            return
        elif best in ("ml_v2", "ML"):
            ml_model = full_comp.get("ml_model", "XGBoost")
            st.success(f"Рекомендуется: ML v2 ({ml_model}){mape_s}")
            return
        elif best in ("econometric", "sarima", "Эконометрика"):
            eco_model = (full_comp.get("eco_model") or full_comp.get("best_eco_model") or "SARIMA")
            eco_label = _ECONOMETRIC_LABELS.get(eco_model, eco_model)
            st.success(f"Рекомендуется: {eco_label}{mape_s}")
            return
        elif best and best not in (None, "None", "—", ""):
            st.info(f"Рекомендуется: {best}{mape_s}")
            return
    if not comp:
        return
    best = comp.get("best_model", "")
    mape = comp.get("best_mape")
    eco_model = comp.get("best_sarima_model", "SARIMA")
    mape_str = f", MAPE = **{mape:.1f}%**" if mape is not None else ""

    if best == "sarima":
        eco_label = _ECONOMETRIC_LABELS.get(eco_model, eco_model)
        st.success(f"Рекомендуется: {eco_label}{mape_str}")
    elif best == "ml_v2":
        winner = comp.get("winner_v2_model", "XGBoost")
        st.success(f"Рекомендуется: ML v2 ({winner} + экзогенные){mape_str}")
    elif best:
        st.info(f"Лучшая модель: {best}{mape_str}")


def _render_comparison_table(available_pairs: list) -> None:
    rows = []
    for pair in available_pairs:
        cat,ch = pair["category"], pair["channel"]
        full = fetch_full_comparison(cat, ch)
        comp = fetch_model_comparison(cat, ch)
        eco_mape= full.get("eco_mape")  or (comp.get("sarima_mape"))
        eco_model = full.get("eco_model") or comp.get("best_sarima_model", "Эконометрика")
        _ml_info_row = fetch_ml_v2_info(cat, ch)
        ml_mape   = (_ml_info_row.get("test_mape_no_exog") or full.get("ml_mape") or comp.get("ml_v2_no_exog"))
        ml_model  = "XGBoost"
        lstm_mape = full.get("lstm_mape")
        best      = full.get("best_overall") or (
            "Эконометрика" if comp.get("best_model") == "sarima" else
            "ML" if comp.get("best_model") == "ml_v2"  else "—")

        rows.append({
            "Категория": cat,
            "Канал": ch,
            f"{eco_model} (тест)": f"{eco_mape:.1f}%" if eco_mape else "—",
            f"ML ({ml_model})":f"{ml_mape:.1f}%"  if ml_mape else "—",
            "LSTM (nо_exog)":f"{lstm_mape:.1f}%" if lstm_mape else "н/д",
            "Лучшая": best})

    if rows:
        df_cmp = pd.DataFrame(rows)
        def highlight_best(row):
            styles = [""]*len(row)
            best_val = row.get("Лучшая", "")
            col_map = {"Эконометрика": 2, "ML": 3, "LSTM": 4}
            idx = col_map.get(best_val)
            if idx is not None and idx < len(styles):
                styles[idx] = "background-color: #d4edda; color: #155724"
            return styles

        st.dataframe(df_cmp.style.apply(highlight_best, axis=1),use_container_width=True, hide_index=True)
        st.caption(
            "MAPE на тестовом периоде (h=12 из одной точки T, no_exog) "
            "Зелёная лучшая модель для ряда")
    else:
        st.info("Данные сравнения моделей временно недоступны.")


def render_aggregate_section(series_list: list[dict], model_class: str,
                               horizon: int) -> None:
    st.divider()
    st.subheader("Агрегированный прогноз")
    st.caption(
        "Суммирует прогнозы по набору рядов."
    )

    agg_model_class = "best"
    st.info(
        "Для каждого ряда автоматически используется лучшая модель "
        "(Эконометрика, ML, LSTM) по результатам тестовой оценки MAPE. "
        "Используемая модель для каждого ряда отображается в таблице после прогноза")

    categories = sorted({s["category"] for s in series_list})
    channels = sorted({s["channel"] for s in series_list})

    col_type, col_vals = st.columns([1,3])
    with col_type:
        agg_type = st.radio(
            "Тип агрегации",
            options=["По категории", "По каналу", "Произвольный набор"],
            key="agg_type_radio")

    with col_vals:
        if agg_type == "По категории":
            selected_cats_agg = st.multiselect(
                "Выберите категории",
                options=categories,
                default=[categories[0]] if categories else [],
                key="agg_cats",
                help="Прогнозы для всех каналов выбранных категорий будут просуммированы")
            agg_pairs = [{"category": s["category"], "channel": s["channel"]}
                for s in series_list
                if s["category"] in selected_cats_agg]
            agg_label = " + ".join(selected_cats_agg) if selected_cats_agg else ""

        elif agg_type == "По каналу":
            selected_chs_agg = st.multiselect(
                "Выберите каналы",
                options=channels,
                default=[channels[0]] if channels else [],
                key="agg_chs",
                help="Прогнозы для всех категорий выбранных каналов будут просуммированы")
            agg_pairs = [
                {"category": s["category"], "channel": s["channel"]}
                for s in series_list
                if s["channel"] in selected_chs_agg]
            agg_label = " + ".join(selected_chs_agg) if selected_chs_agg else ""

        else:
            all_pair_labels = [
                f"{s['category']} | {s['channel']}" for s in series_list]
            selected_labels = st.multiselect(
                "Выберите ряды",
                options=all_pair_labels,
                default=all_pair_labels[:2] if len(all_pair_labels) >= 2 else all_pair_labels,
                key="agg_pairs_custom",)
            label_to_pair = {
                f"{s['category']} | {s['channel']}":
                {"category": s["category"], "channel": s["channel"]}
                for s in series_list}
            agg_pairs = [label_to_pair[l] for l in selected_labels if l in label_to_pair]
            agg_label = "Пользовательский набор"

    if agg_pairs:
        st.caption(f"Рядов для агрегации: **{len(agg_pairs)}**")
    else:
        st.warning("Выберите хотя бы один элемент для агрегации")
        return

    if st.button("Построить агрегированный прогноз", type="secondary",use_container_width=True, key="agg_btn"):
        with st.spinner(f"Строим прогноз для {len(agg_pairs)} рядов"):
            agg_result = fetch_aggregate_forecast(
                pairs=agg_pairs,
                model_class=agg_model_class,
                horizon=horizon,
                label=agg_label)

        if not agg_result or not agg_result.get("forecast"):
            st.error("Не удалось построить агрегированный прогноз")
            return

        hist_vals: dict[str, float] = {}
        for pair in agg_pairs:
            h = fetch_history(pair["category"], pair["channel"])
            if h:
                for date, val in zip(h["dates"], h["values"]):
                    hist_vals[date] = hist_vals.get(date, 0.0) + val
        hist_dates = sorted(hist_vals.keys())
        hist_values = [hist_vals[d] for d in hist_dates]
        history_sum = {"dates": hist_dates, "values": hist_values}

        if agg_result.get("n_errors", 0) > 0:
            st.warning(
                f"Не удалось построить прогноз для {agg_result['n_errors']} "
                f"из {len(agg_pairs)} рядов")

        c1, c2, c3 = st.columns(3)
        c1.metric("Рядов в агрегации", agg_result.get("n_series", len(agg_pairs)))
        total_fc = sum(p["forecast"] for p in agg_result["forecast"])
        c2.metric("Суммарный прогноз (весь горизонт)", f"{total_fc:,.0f} кг")
        c3.metric("Горизонт", f"{horizon} мес.")

        st.plotly_chart(build_aggregate_chart(agg_result, history_sum),use_container_width=True)
        with st.expander("Таблица агрегированного прогноза"):
            fc_df = pd.DataFrame(agg_result["forecast"])
            fc_df.columns = ["Дата", "Прогноз, кг", "Нижн. 80% ДИ", "Верхн. 80% ДИ"]
            for col in ["Прогноз, кг", "Нижн. 80% ДИ", "Верхн. 80% ДИ"]:
                fc_df[col] = fc_df[col].map("{:,.0f}".format)
            st.dataframe(fc_df, use_container_width=True, hide_index=True)
            csv = pd.DataFrame(agg_result["forecast"]).to_csv(index=False).encode("utf-8")
            st.download_button(
                "Скачать CSV",
                csv,
                file_name=f"aggregate_forecast_{agg_label.replace(' ', '_')}.csv",
                mime="text/csv",
                key="agg_download")

        _ind = agg_result.get("individual", [])
        if _ind:
            _cls_map = {"econometric": "Эконометрика", "ml_v2": "ML", "lstm": "LSTM"}
            _cls_counts = {}
            for _it in _ind:
                _k = _cls_map.get(_it.get("model_class", ""), _it.get("model_class", "—"))
                _cls_counts[_k] = _cls_counts.get(_k, 0) + 1
            st.caption(
                "Выбранные модели: "
                + ", ".join(f"{k}: {v} ряд(а)" for k, v in _cls_counts.items()))

        with st.expander("Детализация по рядам"):
            rows = []
            for item in agg_result.get("individual", []):
                fc_pts = item.get("forecast", [])
                total= sum(p.get("forecast", 0) for p in fc_pts)
                mc_raw = item.get("model_class", "")
                _cls_labels = {
                    "econometric": "Эконометрика",
                    "ml_v2":       "ML",
                    "lstm":        "LSTM"}
                winner_model = item.get("winner_model") or item.get("model_type")
                cls_label = _cls_labels.get(mc_raw, mc_raw or "—")
                if winner_model and winner_model != cls_label:
                    display_model = f"{winner_model} ({cls_label})"
                else:
                    display_model = cls_label
                rows.append({
                    "Категория": item["category"],
                    "Канал":item["channel"],
                    "Модель": display_model,
                    "MAPE (тест)": f"{item['test_mape']:.1f}%" if item.get("test_mape") else "—",
                    "Прогноз (сумма), кг": f"{total:,.0f}"})
            if rows:
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def page_analytics(series_list: list[dict]):
    st.title("Аналитика временных рядов")
    categories = sorted({s["category"] for s in series_list})
    channels   = sorted({s["channel"]  for s in series_list})
    col1, col2 = st.columns(2)
    with col1:
        selected_cats = st.multiselect(
            "Категории", categories, default=[categories[0]],
            placeholder="Выберите категории")
    with col2:
        selected_chs = st.multiselect(
            "Каналы", channels, default=[channels[0]],
            placeholder="Выберите каналы")

    if not selected_cats or not selected_chs:
        st.info("Выберите хотя бы одну категорию и один канал")
        return

    available_pairs = [
        s for s in series_list
        if s["category"] in selected_cats and s["channel"] in selected_chs]
    if not available_pairs:
        st.warning("Нет данных для выбранной комбинации")
        return

    with st.spinner("Загружаем данные..."):
        histories = [fetch_history(p["category"], p["channel"]) for p in available_pairs]
        histories = [h for h in histories if h]
    if histories:
        st.plotly_chart(
            build_history_chart(histories, "Исторические продажи"),
            use_container_width=True)

    cat, ch = available_pairs[0]["category"], available_pairs[0]["channel"]
    tabs = st.tabs([
        "STL", "Стационарность", "HP-фильтр", "FFT",
        "Тренд(Mann-Kendall)", "Структурные разрывы",
        "ACF/PACF", "Аномалии",])
    tab_stl, tab_stat, tab_hp, tab_fft, tab_mk, tab_sb, tab_acf, tab_anom = tabs
    with tab_stl:
        st.caption(f"Анализ ряда: **{cat} | {ch}**")
        with st.spinner("STL-декомпозиция"):
            stl = fetch_stl(cat, ch)
        if stl and "error" not in stl:
            m = stl.get("metrics", {})
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Fs(сезонность)",f"{m.get('Fs_seasonal', 0):.3f}",
                      help="0=нет сезонности, 1=сильная")
            c2.metric("Ft(тренд)",f"{m.get('Ft_trend', 0):.3f}",
                      help="0=нет тренда, 1=доминирует")
            c3.metric("R²",f"{m.get('R2', 0):.3f}")
            c4.metric("Амплитуда сез.", f"{m.get('seas_amplitude', 0):,.0f} кг")
            st.plotly_chart(build_stl_chart(stl), use_container_width=True)
            st.info("**Fs > 0.6**: модели с сезонным членом (SARIMA, Prophet) "
                    "**Ft > 0.5**:нужен тренд в модели")
        elif stl.get("error"):
            st.warning(stl["error"])

    with tab_stat:
        st.caption(f"Анализ ряда: **{cat} | {ch}**")
        with st.spinner("Тесты стационарности"):
            stat = fetch_stationarity(cat, ch)
        if stat and stat.get("tests"):
            for test_name, res in stat["tests"].items():
                st.markdown(f"**{test_name}**")
                c1, c2, c3 = st.columns(3)
                c1.metric("Статистика",f"{res.get('adf_stat', res.get('kpss_stat', 0)):.3f}")
                c2.metric("p-value",f"{res.get('adf_p', res.get('kpss_p', 0)):.4f}")
                c3.markdown(
                    f"{'Стационарен' if res.get('is_stationary') else 'Нестационарен'}")
                st.caption(res.get("conclusion", ""))
                st.divider()

    with tab_hp:
        st.caption(f"Анализ ряда: **{cat} | {ch}**")
        st.write(
            "HP-фильтр отделяет долгосрочный тренд от цикла. "
            "Отклонение факта от тренда показывает, в какой фазе находятся продажи: "
            "выше нуля фаза роста, ниже спад или сезонный провал")
        with st.spinner("HP-фильтр"):
            hp = fetch_hp_filter(cat, ch)
        if not hp or "error" in hp:
            st.warning("HP-фильтр недоступен")
        else:
            observed = hp.get("observed", [])
            trend= hp.get("trend", [])
            dates_hp = hp.get("dates", [])
            if observed and trend and dates_hp and len(observed) == len(trend):
                cycle = [float(o) - float(t) for o, t in zip(observed, trend)]
                cycle_pct = [
                    round(c/abs(float(t))*100,1) if float(t) != 0 else 0.0
                    for c, t in zip(cycle, trend)]
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=dates_hp, y=[float(v) for v in observed],
                    mode="lines", name="Факт",
                    line=dict(color="#1f77b4", width=1.5)))
                fig.add_trace(go.Scatter(
                    x=dates_hp, y=[float(v) for v in trend],
                    mode="lines", name="Тренд(HP)",
                    line=dict(color="#d62728", width=2, dash="dash")))
                colors_fill = [
                    "rgba(44,160,44,0.15)"  if c >= 0 else "rgba(214,39,40,0.15)"
                    for c in cycle]
                fig.add_trace(go.Bar(
                    x=dates_hp, y=cycle,
                    name="Цикл",
                    marker_color=["#2ca02c" if c >= 0 else "#d62728" for c in cycle],
                    opacity=0.5,
                    yaxis="y2",
                    hovertemplate="%{x}<br>Цикл: %{y:+,.0f} кг<extra></extra>"))
                fig.update_layout(
                    title="HP-фильтр:тренд, факт и цикл",
                    xaxis_title="Дата",
                    yaxis=dict(title="Объём,кг"),
                    yaxis2=dict(title="Отклонение от тренда,кг",
                                overlaying="y", side="right",
                                showgrid=False, zeroline=True,
                                zerolinecolor="black", zerolinewidth=1),
                    legend=dict(orientation="h", y=-0.2),
                    height=420,
                    bargap=0.0,
                )
                st.plotly_chart(fig, use_container_width=True)
                st.caption(
                    "Синяя линия факт. Красная пунктирная тренд HP "
                    "Столбцы это отклонение от тренда")
            else:
                st.warning("Недостаточно данных для HP-фильтра")

    with tab_fft:
        st.caption(f"Анализ ряда: **{cat} | {ch}**")
        with st.spinner("FFT-анализ"):
            fft = fetch_fft(cat, ch)
        if not fft:
            st.warning("FFT данные не получены. Проверьте API.")
        elif "error" in fft:
            st.warning(f"Ошибка FFT: {fft['error']}")
        else:
            periods = fft.get("top_periods",fft.get("dominant_periods", []))
            freqs = fft.get("all_periods",fft.get("periods", fft.get("freqs", [])))
            spectrum = fft.get("spectrum", fft.get("power",  fft.get("powers", [])))

            if freqs and spectrum and len(freqs) == len(spectrum):
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=freqs, y=spectrum, mode="lines",
                    line=dict(color="#1f77b4", width=1.5),
                    fill="tozeroy", fillcolor="rgba(31,119,180,0.1)",
                    hovertemplate="Период: %{x:.1f} мес.<br>Мощность: %{y:.4f}<extra></extra>"))
                for p in periods[:3]:
                    fig.add_vline(x=p, line_dash="dash", line_color="#d62728",
                                  opacity=0.8,
                                  annotation_text=f"{p:.0f} мес.",
                                  annotation_position="top right")
                fig.update_layout(
                    title="Спектр мощности FFT",
                    xaxis_title="Период (месяцев)",
                    yaxis_title="Мощность",
                    height=370,
                    xaxis=dict(range=[0, max(freqs) if freqs else 60]))
                st.plotly_chart(fig, use_container_width=True)
            elif freqs or spectrum:
                st.warning(
                    f"Данные FFT получены, но размерности не совпадают: "
                    f"periods={len(freqs)}, spectrum={len(spectrum)}. "
                    f"Ключи ответа: {list(fft.keys())}")
            else:
                st.info(
                    f"FFT ответ получен, но нет данных для графика. "
                    f"Ключи: {list(fft.keys())}")
                if fft:
                    with st.expander("Сырой ответ API"):
                        st.json({k: str(v)[:200] for k, v in fft.items()})

            if periods:
                c1, c2 = st.columns(2)
                c1.write("**Доминирующие периоды:**")
                for i, p in enumerate(periods[:3], 1):
                    c1.write(f"  {i}. {p:.1f} месяцев")
                c2.write("**Интерпретация:**")
                has_12 = any(abs(p - 12) < 2 for p in periods[:3])
                has_6  = any(abs(p - 6)  < 1.5 for p in periods[:3])
                c2.write("Пик ~12 мес. годовая сезонность" if has_12
                         else "Нет выраженной годовой сезонности")
                if has_6:
                    c2.write("Пик ~6 мес. полугодовой цикл")

    with tab_mk:
        st.caption(f"Анализ ряда: **{cat} | {ch}**")
        with st.spinner("Mann-Kendall тест"):
            mk = fetch_mann_kendall(cat, ch)
        if not mk:
            st.warning("Данные теста недоступны")
        elif "error" in mk:
            st.warning(f"Ошибка: {mk['error']}")
        else:
            trend = (mk.get("trend") or mk.get("trend_direction") or mk.get("direction") or "—")
            p_val = (mk.get("p") or mk.get("p_value") or mk.get("pvalue") or mk.get("p_mk") or 0.0)
            slope = (mk.get("sens_slope_per_month") or mk.get("slope_per_month") or mk.get("sens_slope") or mk.get("slope") or 0.0)
            sig = (mk.get("significant") or mk.get("trend_significant") or (float(p_val) < 0.05 if p_val else False))
            c1,c2,c3 = st.columns(3)
            c1.metric("Направление тренда",str(trend) if trend != "—" else "—")
            c2.metric("p-value",f"{float(p_val):.4f}" if p_val else "—")
            slope_yr = mk.get("sens_slope_per_year")
            slope_lbl = f"{float(slope):+.1f} кг/мес"
            if slope_yr:
                slope_lbl += f" ({float(slope_yr):+.0f} кг/год)"
            c3.metric("Наклон Сена",slope_lbl if slope else "—",
                help="Наклон Сена показывает медианное изменение за месяц/год")

            if trend and trend != "—":
                if sig:
                    st.success(f"Тренд значим(p<0.05):{trend}")
                else:
                    st.info(f"Тренд не значим(p>=0.05):{trend}")
            else:
                st.info("Направление тренда не определено")



    with tab_sb:
        st.caption(f"Анализ ряда: **{cat} | {ch}**")
        st.write(
            "Тест Bai-Perron выявляет точки, где статистически значимо "
            "изменяется средний уровень или динамика ряда. "
            "Если разрывов нет то ряд статистически однородный, "
            "без выраженных структурных сдвигов")
        with st.spinner("Структурные разрывы"):
            sb = fetch_structural_breaks(cat, ch)
        if not sb or "error" in sb:
            st.warning("Данные недоступны")
        else:
            breaks = sb.get("breakpoints", [])
            if breaks:
                st.warning(f"Найдено {len(breaks)} разрывов: {', '.join(str(b) for b in breaks)}")
            else:
                st.write("Структурных разрывов не обнаружено")
            dates_hist = histories[0]["dates"] if histories else []
            vals_hist  = histories[0]["values"] if histories else []
            if dates_hist:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=dates_hist, y=vals_hist, mode="lines",line=dict(color="#1f77b4")))
                for bp in breaks:
                    fig.add_vline(x=bp, line_dash="dash", line_color="red", opacity=0.7)
                fig.update_layout(title="Структурные разрывы", height=350)
                st.plotly_chart(fig, use_container_width=True)

    with tab_acf:
        diff_option = st.selectbox(
            "Порядок разности",
            options=["original", "d=1", "D=1", "d=1+D=1"],
            format_func=lambda x: {
                "original": "Исходный ряд",
                "d=1": "d=1 (первая разность)",
                "D=1":"D=1 (сезонная разность)",
                "d=1+D=1": "d=1 + D=1"}.get(x, x))
        with st.spinner("Считаем ACF/PACF"):
            acf_data = fetch_acf_pacf(cat, ch, diff=diff_option)
        if acf_data and "error" not in acf_data:
            st.plotly_chart(build_acf_pacf_chart(acf_data), use_container_width=True)
            st.caption(f"Доверительная граница {acf_data['conf_bound']:.3f} (95%)")
            st.info("**Медленно убывающий ACF**: AR-компонент(p) "
                    "**Обрыв PACF на k**: порядок p=k "
                    "**Пики на 12, 24**: сезонность")

    with tab_anom:
        method = st.selectbox(
            "Метод выявления аномалий",
            ["stl", "iqr", "zscore"],
            format_func=lambda x: {
                "stl":"STL-остатки",
                "iqr":"IQR (межквартильный размах)",
                "zscore":"Z-score"}.get(x, x))
        with st.spinner("Ищем аномалии"):
            anom = fetch_anomalies(cat, ch, method=method)
        if anom:
            c1, c2, c3 = st.columns(3)
            c1.metric("Аномалий найдено",anom["n_anomalies"])
            c2.metric("Доля аномалий",f"{anom['pct_anomalies']:.1f}%")
            c3.metric("Из них в COVID-период",len(anom.get("covid_anomalies", [])))
            st.caption(anom.get("interpretation", ""))
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=anom["dates"], y=anom["values"], mode="lines", name="Факт",
                line=dict(color="#1f77b4", width=1.5)))
            if anom["anomaly_dates"]:
                colors = ["#d62728" if t == "низкий" else "#ff7f0e" for t in anom["anomaly_types"]]
                fig.add_trace(go.Scatter(
                    x=anom["anomaly_dates"], y=anom["anomaly_values"],
                    mode="markers", name="Аномалия",
                    marker=dict(color=colors, size=10, symbol="x")))
            fig.update_layout(title=f"Аномалии ({method})", xaxis_title="Дата", yaxis_title="Объём, кг", height=380)
            st.plotly_chart(fig, use_container_width=True)
            st.info("**Оранжевые** неожиданно высокие "
                    "**Красные** неожиданно низкие")

    st.divider()
    st.subheader("Сводный анализ по всем рядам")
    summary_tab, covid_tab, corr_tab = st.tabs([
        "Тренд и сезонность", "COVID-влияние", "Корреляции между рядами"])

    with summary_tab:
        with st.spinner("Загружаем сводную аналитику"):
            summary = fetch_analytics_summary()
        if summary and summary.get("summary"):
            df_sum = pd.DataFrame(summary["summary"]).rename(columns={
                "category":"Категория",
                "channel":"Канал",
                "n_obs":"Наблюд.",
                "mean": "Среднее, кг",
                "trend_direction": "Тренд",
                "trend_significant":"Тренд значим",
                "sens_slope_year":"Наклон кг/год",
                "seasonality":"Сезонность",
                "seasonality_significant": "Сез. значима",
                "eta_squared":"Eta^2"})
            for col in ["Тренд значим", "Сез. значима"]:
                if col in df_sum.columns:
                    df_sum[col] = df_sum[col].map({True: "Значим", False: "Не значим"})
            st.dataframe(
                df_sum.style.map(
                    lambda v: "color: green" if v == "Значим" else "color: red"   if v == "Не значим" else "",
                    subset=[c for c in ["Тренд значим", "Сез. значима"] if c in df_sum.columns]),
                use_container_width=True, hide_index=True)

    with covid_tab:
        channel_filter = st.selectbox(
            "Фильтр по каналу", ["Все каналы"] + sorted(channels), key="covid_ch")
        ch_param = None if channel_filter == "Все каналы" else channel_filter
        with st.spinner("Считаем COVID-влияние"):
            covid = fetch_covid_impact(ch_param)
        if covid:
            df_covid = pd.DataFrame(covid).rename(columns={
                "category":"Категория",
                "channel":"Канал",
                "covid_mean":"Ср. в COVID, кг",
                "pre_covid_mean": "Ср. 2019 (тот же пер.), кг",
                "drop_yoy_pct":"Падение YoY %",
                "impact_level":"Уровень удара"})
            st.dataframe(df_covid, use_container_width=True, hide_index=True)
            df_plot = pd.DataFrame(covid).dropna(subset=["drop_yoy_pct"])
            if not df_plot.empty:
                df_plot["label"] = df_plot["category"] + " | " + df_plot["channel"]
                df_plot = df_plot.sort_values("drop_yoy_pct")
                fig = go.Figure(go.Bar(
                    x=df_plot["drop_yoy_pct"], y=df_plot["label"], orientation="h",
                    marker_color=["#d62728" if v < 0 else "#2ca02c" for v in df_plot["drop_yoy_pct"]],
                    hovertemplate="%{y}<br>YoY: %{x:.1f}%<extra></extra>"))
                fig.add_vline(x=0, line_color="black", line_width=1)
                fig.update_layout(
                    title="Изменение продаж в COVIDvs2019 %",
                    xaxis_title="Изменение %",
                    height=max(400, len(df_plot) * 22),
                    margin=dict(l=250))
                st.plotly_chart(fig, use_container_width=True)

    with corr_tab:
        ch_param_c = None
        channel_filter_c = st.selectbox("Фильтр по каналу", ["Все каналы"] + sorted(channels), key="corr_ch")
        ch_param_c = None if channel_filter_c == "Все каналы" else channel_filter_c
        with st.spinner("Считаем корреляции"):
            corr = fetch_correlation(ch_param_c)
        if corr and "error" not in corr:
            labels = corr["labels"]
            matrix = corr["matrix"]
            c1, c2 = st.columns(2)
            c1.metric("Рядов в анализе",corr["n_series"])
            c2.metric("Средняя корреляция",f"{corr['mean_correlation']:.3f}")
            fig = go.Figure(go.Heatmap(
                z=matrix, x=labels, y=labels,
                colorscale="RdYlGn", zmid=0, zmin=-1, zmax=1,
                text=[[f"{v:.2f}" for v in row] for row in matrix],
                texttemplate="%{text}",
                hovertemplate="%{y} × %{x}<br>r = %{z:.3f}<extra></extra>"))
            fig.update_layout(
                title="Матрица корреляций",
                height=max(400, len(labels) * 40 + 100),
                xaxis=dict(tickangle=30))
            st.plotly_chart(fig, use_container_width=True)
            col_top, col_bot = st.columns(2)
            with col_top:
                st.markdown("**Наиболее коррелированные:**")
                for p in corr.get("top_pairs", [])[:5]:
                    st.markdown(f"{p['series_a']}X{p['series_b']}: **r={p['r']:.3f}**")
            with col_bot:
                st.markdown("**Наименее коррелированные:**")
                for p in corr.get("bottom_pairs", [])[:3]:
                    st.markdown(f"{p['series_a']}X{p['series_b']}: **r={p['r']:.3f}**")

def page_forecast(series_list: list[dict]):
    st.title("Прогнозирование продаж")
    categories = sorted({s["category"] for s in series_list})
    channels = sorted({s["channel"]  for s in series_list})
    st.subheader("Параметры")
    col1,col2,col3 = st.columns([2, 2, 1])
    with col1:
        selected_cats = st.multiselect(
            "Категории", categories, default=[categories[0]],
            placeholder="Выберите категории...", key="fc_cats")
    with col2:
        selected_chs = st.multiselect(
            "Каналы", channels, default=[channels[0]],
            placeholder="Выберите каналы...", key="fc_chs")
    with col3:
        horizon = st.number_input("Горизонт (мес)", min_value=1, max_value=60, value=12, step=1)

    if not selected_cats or not selected_chs:
        st.info("Выберите хотя бы одну категорию и один канал")
        return

    available_pairs = [s for s in series_list if s["category"] in selected_cats and s["channel"] in selected_chs]
    if not available_pairs:
        st.warning("Нет данных для выбранной комбинации")
        return

    st.caption(f"Выбрано рядов: **{len(available_pairs)}**")

    full_comp0 = {}
    comp0 = {}
    if len(available_pairs)==1:
        cat0, ch0 = available_pairs[0]["category"], available_pairs[0]["channel"]
        full_comp0 = fetch_full_comparison(cat0, ch0)
        comp0 = fetch_model_comparison(cat0, ch0)
        _render_model_badge(comp0, full_comp=full_comp0)

    st.subheader("Модель")

    lstm_variants = fetch_lstm_variants()
    lstm_available = len(lstm_variants) > 0

    model_options = ["econometric", "ml_v2"]
    if lstm_available:
        model_options.append("lstm")
    model_class = st.radio(
        "Класс модели",
        options=model_options,
        format_func=lambda x: {
            "econometric": "Эконометрика (Naive / ARIMA / SARIMA / SARIMAX / Prophet)",
            "ml_v2": "ML (XGBoost)",
            "lstm": f"LSTM",
        }.get(x, x),
        horizontal=True,
        key="model_class_radio")

    if not lstm_available and "lstm" in (model_class or ""):
        st.warning("LSTM модели не обучены. Запустите: `python -m src.training.train_lstm`")
        return
    exog_file = None
    user_exog_kv = None

    _SARIMA_EXOG_VARS = [
        ("NT_Price per kg", "Цена за кг, руб."),
        ("Penetration", "Пенетрация, %"),
        ("MT_Universe percent", "Доля отгруженных магазинов"),
        ("Frequency", "Частота покупок")]

    if model_class == "econometric":
        with st.expander("Плановые экзогенные переменные для SARIMAX (необязательно)"):
            st.markdown(
                "Если известны плановые значения введите вручную или загрузите файл "
                "Без данных модель прогнозирует значение экзогенной отдельной SARIMA моделью"
                "(актуально только если будет строиться SARIMAX модель)"
            )

            input_mode = st.radio(
                "Способ ввода",
                ["Ручной ввод", "Загрузить файл"],
                horizontal=True,
                key="eco_exog_mode")

            if input_mode == "Ручной ввод":
                st.caption(
                    f"Введите {horizon} "
                    f"значений через запятую."
                )
                eco_exog_kv = {}
                for col, label in _SARIMA_EXOG_VARS:
                    raw = st.text_input(
                        label,
                        placeholder=f"Через запятую, {horizon} значений",
                        key=f"eco_exog_{col}")
                    if raw.strip():
                        try:
                            vals = [float(v.strip()) for v in raw.split(",") if v.strip()]
                            if vals:
                                eco_exog_kv[col] = vals if len(vals) > 1 \
                                                   else [vals[0]] * horizon
                        except ValueError:
                            st.warning(f"Некорректные значения для «{label}»")

                if eco_exog_kv:
                    import io
                    eco_df = pd.DataFrame(eco_exog_kv)
                    buf = io.BytesIO()
                    eco_df.to_csv(buf, index=False)
                    buf.seek(0)
                    class _PseudoFile:
                        def __init__(self, b, name):
                            self._b = b; self.name = name
                        def getvalue(self): return self._b.getvalue()
                        def seek(self, p): self._b.seek(p)
                    exog_file = _PseudoFile(buf, "eco_exog.csv")

            else:  # Загрузить файл
                st.markdown(
                    "**Формат файла** (Excel или CSV):\n"
                    "| NT_Price per kg | Penetration | MT_Universe percent | Frequency |\n"
                    "|---|---|---|---|\n"
                    f"Количество строк=горизонт ({horizon} мес.)"
                )
                exog_file = st.file_uploader(
                    "Файл Excel или CSV",
                    type=["xlsx", "xls", "csv"],
                    key="exog_upload_eco")
                if exog_file:
                    try:
                        preview = (pd.read_excel(exog_file)
                                   if exog_file.name.endswith((".xlsx", ".xls"))
                                   else pd.read_csv(exog_file))
                        st.success(f"Файл загружен: {len(preview)} строк")
                        st.dataframe(preview.head(3), use_container_width=True)
                        exog_file.seek(0)
                    except Exception as e:
                        st.error(f"Ошибка чтения: {e}")
                        exog_file = None

    elif model_class == "ml_v2":
        ml_info = {}
        if len(available_pairs) == 1:
            ml_info = fetch_ml_v2_info(
                available_pairs[0]["category"], available_pairs[0]["channel"])
        selected_exog_vars = ml_info.get("selected_exog", [])

        with st.expander("Информация о ML модели (XGBoost vif_exog)"):
            c1, c2 = st.columns(2)
            c1.write("**Алгоритм:** XGBoost")
            c1.write("**Конфигурация:** vif_exog")
            _xgb_mape_no   = ml_info.get("test_mape_no_exog")
            _xgb_mape_full = ml_info.get("test_mape_full_exog")
            if _xgb_mape_no is not None:
                c2.metric("MAPE (no_exog)",
                    f"{_xgb_mape_no:.2f}%")
            if _xgb_mape_full is not None:
                c2.metric(
                    "MAPE (full_exog, этот ряд)",
                    f"{_xgb_mape_full:.2f}%",
                    help="MAPE с реальными тестовыми экзогенными")
            st.caption(
                "Одна глобальная модель для всех 44 рядов. ")

        with st.expander("Плановые значения переменных (необязательно)", expanded=False):
            st.markdown(
                "Введите плановые значения переменных для повышения точности. "
                "Пустые поля вызывают режим **no_exog** (последнее известное значение)"
            )
            if not selected_exog_vars:
                st.info("Список переменных модели временно недоступен.")
                _default_vars = [
                    "NT_Price per kg", "Penetration",
                    "NT_CWD", "Spend per Trip", "Volume per Trip"]
                st.markdown(
                    "Ожидаемый формат файла (Excel или CSV): колонки = "
                    "названия экзогенных переменных, строк=горизонт прогноза.\n"
                    "Возможные колонки: "
                    + ", ".join(f"`{c}`" for c in _default_vars))
                exog_file = st.file_uploader(
                    "Загрузить файл (Excel или CSV)",
                    type=["xlsx", "xls", "csv"],
                    key="exog_upload_ml_fallback")
            else:
                input_mode = st.radio(
                    "Способ ввода", ["Ввести вручную", "Загрузить файл"],
                    horizontal=True, key="exog_input_mode")
                if input_mode == "Загрузить файл":
                    st.markdown("**Формат файла** (Excel или CSV):")
                    _example = {col: [f"знач_{h+1}" for h in range(min(3, horizon))] for col in selected_exog_vars}
                    st.dataframe(
                        pd.DataFrame(_example,
                                     index=[f"Месяц {h+1}" for h in range(min(3, horizon))]),
                        use_container_width=True)
                    st.caption(
                        f"Количество строк=горизонт прогноза ({horizon} мес.) "
                        f"Названия колонок должны точно совпадать: "
                        + ", ".join(f"`{c}`" for c in selected_exog_vars))

                    import io as _io
                    _last_data_date = None
                    if len(available_pairs) >= 1:
                        try:
                            _hist = fetch_history(
                                available_pairs[0]["category"],
                                available_pairs[0]["channel"])
                            _dates = _hist.get("dates", [])
                            if _dates:
                                _last_data_date = pd.Timestamp(_dates[-1])
                        except Exception:
                            pass
                    if _last_data_date is None:
                        try:
                            from src.config.settings import DATA_PATH
                            from src.data.load_data import load_data
                            _df_tmp = load_data(str(DATA_PATH))
                            _last_data_date = pd.Timestamp(_df_tmp.index.max())
                        except Exception:
                            _last_data_date = pd.Timestamp("2024-05-01")
                    _tmpl_start = _last_data_date + pd.DateOffset(months=1)
                    _tmpl = pd.DataFrame(
                        {col: [None] * horizon for col in selected_exog_vars},
                        index=pd.date_range(
                            start=_tmpl_start,
                            periods=horizon,
                            freq="MS").strftime("%Y-%m-%d"))
                    _buf = _io.BytesIO()
                    _tmpl.to_csv(_buf)
                    _buf.seek(0)
                    st.download_button(
                        label="Скачать шаблон CSV",
                        data=_buf,
                        file_name="ml_exog_template.csv",
                        mime="text/csv",
                        help="Заполните шаблон плановыми значениями и загрузите обратно")

                    exog_file = st.file_uploader(
                        "Загрузить заполненный файл",
                        type=["xlsx", "xls", "csv"],
                        key="exog_upload_ml")
                    if exog_file:
                        try:
                            if exog_file.name.endswith((".xlsx", ".xls")):
                                preview = pd.read_excel(exog_file)
                            else:
                                import io as _csv_io
                                _raw = exog_file.read()
                                exog_file.seek(0)
                                _sample = _raw[:2048].decode("utf-8", errors="ignore")
                                _sep = ";" if _sample.count(";") > _sample.count(",") else ","
                                preview = pd.read_csv(_csv_io.BytesIO(_raw),sep=_sep,index_col=0)
                            missing = [c for c in selected_exog_vars if c not in preview.columns]
                            extra   = [c for c in preview.columns if c not in selected_exog_vars]
                            if missing:
                                st.warning(
                                    f"Отсутствуют колонки: {missing}. "
                                    f"Они будут заполнены последним известным значением")
                            if extra:
                                st.info(f"Лишние колонки будут проигнорированы: {extra}")
                            st.success(
                                f"Файл загружен: {len(preview)} строк Х  "
                                f"{len(preview.columns)} колонок")
                            st.dataframe(preview, use_container_width=True)
                            exog_file.seek(0)
                        except Exception as e:
                            st.error(f"Ошибка чтения файла: {e}")
                            exog_file = None
                else:
                    user_exog_kv = {}
                    for col in selected_exog_vars:
                        label = _ML_V2_EXOG_LABELS.get(col, col)
                        raw = st.text_input(label,
                            placeholder=f"Через запятую, {horizon} значений",
                            key=f"exog_input_{col}")
                        if raw.strip():
                            try:
                                vals = [float(v.strip()) for v in raw.split(",")
                                        if v.strip()]
                                if vals:
                                    user_exog_kv[col] = [float(v) for v in vals]
                            except ValueError:
                                st.warning(f"Некорректные значения для «{label}»")
                    st.caption(f"Заполнено: {len(user_exog_kv)} из {len(selected_exog_vars)} переменных"
                        if user_exog_kv else "Режим no_exog (без плановых данных)")

    elif model_class == "lstm":
        user_exog_kv = None
        lstm_info = fetch_lstm_info("lstm_exog") if lstm_variants else {}
        has_future = lstm_info.get("has_future_exog", False)
        lstm_exog_vars = lstm_info.get("vif_exog_vars", []) if has_future else []
        with st.expander("Информация о LSTM модели"):
            st.write("Продакшн модель: **LSTM (exog)**")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Архитектура", lstm_info.get("model_class", "LSTM"))
            c2.metric("Hidden size",lstm_info.get("hidden_size", "-"))
            c3.metric("Window size", lstm_info.get("window_size", "-"))
            c4.metric("Признаков", lstm_info.get("n_features", "-"))
            _series_mape = None
            if len(available_pairs) == 1:
                _series_mape = full_comp0.get("lstm_mape") if full_comp0 else None
                if _series_mape is None:
                    try:
                        from src.config.settings import ML_RESULTS_DIR as _r
                        _lpath = _r.parent / "part3" / "lstm_exog_per_series.csv"
                        if _lpath.exists():
                            _ldf = pd.read_csv(_lpath)
                            _lrow = _ldf[
                                (_ldf["category"] == available_pairs[0]["category"])&
                                (_ldf["channel"]  == available_pairs[0]["channel"])]
                            if not _lrow.empty:
                                for _mc in ["mape_no_exog", "lstm_mape_no_exog", "mape"]:
                                    if _mc in _lrow.columns:
                                        _series_mape = float(_lrow.iloc[0][_mc])
                                        break
                    except Exception:
                        pass

            if _series_mape is not None:
                st.metric(
                    "MAPE на тесте(этот ряд)",
                    f"{float(_series_mape):.2f}%",
                    help="Тестовый MAPE модели lstm_exog")
            elif lstm_info.get("mape_no_exog"):
                st.metric(
                    "MAPE (медиана, все ряды)",
                    f"{lstm_info['mape_no_exog']:.2f}%",
                    help="Per-series данные недоступны,медиана по всем 44 рядам")


        if has_future and lstm_exog_vars:
            with st.expander("Плановые значения переменных для LSTM (необязательно)"):
                st.markdown(
                    "LSTM с экзогенными (lstm_exog) может использовать плановые значения. "
                    "Оставьте пустым, используется последнее известное значение (no_exog).")
                user_exog_kv = {}
                for col in lstm_exog_vars:
                    label = _ML_V2_EXOG_LABELS.get(col, col)
                    raw = st.text_input(
                        label,
                        placeholder=f"1 число или несколько через запятую",
                        key=f"lstm_exog_{col}")
                    if raw.strip():
                        try:
                            vals = [float(v.strip()) for v in raw.split(",")
                                    if v.strip()]
                            if vals:
                                user_exog_kv[col] = [float(v) for v in vals]
                        except ValueError:
                            st.warning(f"Некорректные значения для «{label}»")
                if not user_exog_kv:
                    st.caption("Режим no_exog, плановые данные не переданы")

    with st.expander("Сравнение всех моделей (MAPE на тестовом периоде)"):
        _render_comparison_table(available_pairs)

    if st.button("Построить прогноз", type="primary", use_container_width=True):
        forecasts = []
        histories = []
        errors = []

        progress = st.progress(0, text="Строим прогнозы...")
        for idx, pair in enumerate(available_pairs):
            cat, ch = pair["category"], pair["channel"]
            h = fetch_history(cat, ch)
            if h:
                histories.append(h)

            if model_class == "lstm":
                fc = fetch_forecast_lstm(cat, ch, horizon,user_exog=user_exog_kv if user_exog_kv else None)
            elif model_class == "ml_v2":
                fc = fetch_forecast(cat, ch, horizon,model_class="ml_v2",
                    user_exog=user_exog_kv if user_exog_kv else None,
                    exog_file=exog_file)
            else:
                fc = fetch_forecast(cat, ch, horizon, model_class="econometric",exog_file=exog_file)
            if fc:
                forecasts.append(fc)
            else:
                errors.append(f"{cat} | {ch}")
            if exog_file:
                exog_file.seek(0)
            progress.progress((idx + 1) / len(available_pairs),text=f"Прогноз: {cat} | {ch}")
        progress.empty()

        if errors:
            st.warning(f"Не удалось: {', '.join(errors)}")
        if not forecasts:
            st.error("Ни одного прогноза не получено.")
            st.session_state["_last_forecasts"] = []
            return

        st.session_state["_last_forecasts"] = forecasts
        st.session_state["_last_histories"] = histories
        st.session_state["_last_model_class"] = model_class
        st.session_state["_last_horizon"]= horizon

        st.divider()
        if len(forecasts) == 1:
            fc = forecasts[0]
            h = histories[0] if histories else {}
            c1,c2,c3 = st.columns(3)
            c1.metric("Модель",fc.get("model_type", "—"))
            c2.metric("Горизонт", f"{horizon} мес.")
            c3.metric("MAPE (тест h=1..12)",
                      f"{fc['test_mape']:.1f}%" if fc.get("test_mape") else "—",
                      help="Медиана MAPE на тестовом периоде")
            if fc.get("exog_cols"):
                st.caption(f"Переданы плановые данные: {', '.join(fc['exog_cols'])}")
            elif model_class == "ml_v2":
                st.caption("Режим без плановых данных (no_exog)")
            st.plotly_chart(build_forecast_chart(h, fc), use_container_width=True)

            with st.expander("Таблица прогноза"):
                fc_df = pd.DataFrame(fc["forecast"])
                fc_df.columns = ["Дата", "Прогноз,кг", "Нижн.80% ДИ", "Верхн.80% ДИ"]
                for col in ["Прогноз,кг", "Нижн.80% ДИ", "Верхн.80% ДИ"]:
                    fc_df[col] = fc_df[col].map("{:,.0f}".format)
                st.dataframe(fc_df, use_container_width=True, hide_index=True)
                csv = pd.DataFrame(fc["forecast"]).to_csv(index=False).encode("utf-8")
                st.download_button("Скачать CSV", csv,
                    file_name=f"forecast_{fc['category']}_{fc['channel']}.csv",
                    mime="text/csv")

        else:
            tabs_labels = ([f"{fc['category']} | {fc['channel']}" for fc in forecasts]+["Суммарно"])
            tabs = st.tabs(tabs_labels)
            for tab,fc,h in zip(tabs[:-1], forecasts, histories):
                with tab:
                    col1,col2 = st.columns(2)
                    col1.metric("Модель", fc.get("model_type", "—"))
                    col2.metric("MAPE (тест)",f"{fc['test_mape']:.1f}%" if fc.get("test_mape") else "-")
                    st.plotly_chart(build_forecast_chart(h, fc), use_container_width=True)
                    with st.expander("Таблица"):
                        fc_df = pd.DataFrame(fc["forecast"])
                        st.dataframe(fc_df, use_container_width=True, hide_index=True)
                        csv = fc_df.to_csv(index=False).encode("utf-8")
                        st.download_button("CSV", csv,
                            file_name=f"forecast_{fc['category']}_{fc['channel']}.csv",
                            mime="text/csv", key=f"dl_{fc['category']}_{fc['channel']}")

            with tabs[-1]:
                fc_df_all = pd.concat([pd.DataFrame(fc["forecast"]) for fc in forecasts])
                agg_df = fc_df_all.groupby("date").agg(
                    forecast=("forecast", "sum"),
                    lower_80=("lower_80", "sum"),
                    upper_80=("upper_80", "sum")).reset_index()
                agg_result_inline = {
                    "label":f"{', '.join(selected_cats)} | {', '.join(selected_chs)}",
                    "forecast": agg_df.to_dict("records")}
                hist_sum_inline = None
                if histories:
                    hdf = pd.concat([
                        pd.DataFrame({"date": h["dates"], "value": h["values"]})
                        for h in histories])
                    hagg = hdf.groupby("date")["value"].sum().reset_index()
                    hist_sum_inline = {
                        "dates":  hagg["date"].tolist(),
                        "values": hagg["value"].tolist()}
                st.plotly_chart(
                    build_aggregate_chart(agg_result_inline, hist_sum_inline),
                    use_container_width=True)
                with st.expander("Суммарная таблица"):
                    agg_display = agg_df.copy()
                    agg_display.columns = ["Дата", "Прогноз,кг", "Нижн.80% ДИ", "Верхн.80% ДИ"]
                    st.dataframe(agg_display, use_container_width=True, hide_index=True)
                    csv = agg_df.to_csv(index=False).encode("utf-8")
                    st.download_button("Скачать суммарный CSV", csv,
                        file_name="forecast_aggregate.csv", mime="text/csv")

    render_aggregate_section(
        series_list=series_list,
        model_class=model_class if "model_class" in dir() else "econometric",
        horizon=horizon if "horizon" in dir() else 12)

def main():
    st.sidebar.title("Аналитика и прогноз продаж Mars")
    st.sidebar.divider()
    page = st.sidebar.radio("Раздел",["Аналитика", "Прогноз"],label_visibility="collapsed")

    st.sidebar.divider()
    st.sidebar.caption("API: " +API_URL)

    try:
        r = requests.get(f"{API_URL}/health", timeout=3)
        if r.status_code == 200:
            st.sidebar.success("API доступен")
        else:
            st.sidebar.error("API недоступен")
    except Exception:
        st.sidebar.error("API недоступен")
        st.error("Не удалось подключиться к API. uvicorn src.api.main:app --port 8000")
        return

    series_list = fetch_series()
    if not series_list:
        st.error("Нет данных. Проверьте подключение к API.")
        return

    st.sidebar.caption(f"Рядов в базе: {len(series_list)}")

    if page == "Аналитика":
        page_analytics(series_list)
    elif page == "Прогноз":
        page_forecast(series_list)


if __name__ == "__main__":
    main()