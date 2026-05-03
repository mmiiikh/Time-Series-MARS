import os
import asyncio
import pandas as pd
from io import BytesIO
from typing import List, Optional
from fastapi import APIRouter, HTTPException, UploadFile, File, Body

from src.api.schemas import (
    SeriesInfo, HistoryResponse,
    StlResponse, StationarityResponse, AcfPacfResponse,
    ForecastRequest, ForecastResponse, ForecastPoint,
    ModelInfo, ModelComparisonInfo,
    TrainRequest, TrainResponse,
    AggregateRequest, AggregateResponse)
from src.data.load_data import load_data, create_series_dict
from src.data.preprocess import (
    get_series_stats, get_series_history,
    get_stl_decomposition, get_stationarity, get_acf_pacf)
from src.forecasting.sarima import forecast_from_manifest, load_manifest
from src.config.settings import DATA_PATH, ML_DATA_FILE, ML_RESULTS_DIR

from src.data.analytics import (
    get_hp_filter, get_fft_analysis, get_mann_kendall,
    get_structural_breaks, get_seasonal_subseries,
    get_seasonality_test, get_anomalies, get_covid_impact,
    get_cross_series_correlation, get_cross_correlation,
    get_full_analytics_summary)

router = APIRouter()

_df = load_data(str(DATA_PATH))
_series_dict = create_series_dict(_df)

try:
    _df_ml = load_data(str(ML_DATA_FILE))
    _series_dict_ml = create_series_dict(_df_ml)
except Exception as _e:
    _series_dict_ml = _series_dict

TRAIN_SECRET = os.getenv("TRAIN_SECRET_KEY", "mars-train-2024")
_ALLOWED_MODEL_CLASSES = {"econometric", "ml_v2", "lstm", "best"}

@router.get("/series", response_model=list[SeriesInfo], tags=["Series"])
def get_series():
    return get_series_stats(_series_dict)

@router.get("/series/{category}/{channel}/history",
            response_model=HistoryResponse, tags=["Series"])
def get_history(category: str, channel: str):
    result = get_series_history(_series_dict, category, channel)
    if not result:
        raise HTTPException(status_code=404, detail=f"Ряд '{category}|{channel}' не найден")
    return result


@router.post("/forecast", response_model=ForecastResponse, tags=["Forecast"])
async def forecast(request: ForecastRequest):
    if request.model_class not in _ALLOWED_MODEL_CLASSES:
        raise HTTPException(status_code=400,detail=f"model_class должен быть: {' | '.join(_ALLOWED_MODEL_CLASSES)}")

    if request.model_class == "lstm":
        try:
            from src.forecasting.lstm_forecast import forecast_lstm, is_lstm_available
            if not is_lstm_available():
                raise HTTPException(status_code=404,detail="LSTM модели не обучены. Запустите src/training/train_lstm.py")
            from src.forecasting.lstm_forecast import PROD_VARIANT
            result = await asyncio.to_thread(
                forecast_lstm,
                _series_dict_ml,
                request.category,
                request.channel,
                request.horizon,
                PROD_VARIANT,
                request.exog_data)
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Ошибка LSTM прогноза: {e}")

        return ForecastResponse(
            category=result["category"],
            channel=result["channel"],
            model_type=result["model_type"],
            model_spec=result["model_spec"],
            model_class="lstm",
            exog_cols=result.get("exog_cols", []),
            test_mape=result.get("test_mape"),
            horizon=request.horizon,
            forecast=[ForecastPoint(**p) for p in result["forecast"]])

    if request.model_class == "ml_v2":
        try:
            from src.forecasting.ml_v2_forecast import forecast_ml_v2
            user_exog = None
            if request.exog_data:
                user_exog = {}
                for col, val in request.exog_data.items():
                    if isinstance(val, list):
                        user_exog[col] = [float(v) for v in val]
                    else:
                        user_exog[col] = [float(val)] * request.horizon

            result = forecast_ml_v2(
                series_dict=_series_dict,
                category=request.category,
                channel=request.channel,
                horizon=request.horizon,
                user_exog=user_exog)
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            import traceback as _tb
            raise HTTPException(status_code=500,detail=f"Ошибка ML v2 прогноза: {e}\n{_tb.format_exc()}")

        return ForecastResponse(
            category=result["category"],
            channel=result["channel"],
            model_type=result["model_type"],
            model_spec=result["model_spec"],
            model_class="ml_v2",
            exog_cols=result["exog_cols"],
            test_mape=result.get("test_mape"),
            horizon=request.horizon,
            forecast=[ForecastPoint(**p) for p in result["forecast"]])

    exog_df = None
    if request.exog_data:
        try:
            exog_df = pd.DataFrame(request.exog_data)
        except Exception as e:
            raise HTTPException(status_code=422, detail=f"Ошибка в exog_data: {e}")

    try:
        result = forecast_from_manifest(
            series_dict=_series_dict,
            category=request.category,
            channel=request.channel,
            horizon=request.horizon,
            exog_df=exog_df)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка прогноза: {e}")

    try:
        manifest = load_manifest()
        key = f"{request.category}|{request.channel}"
        test_mape = manifest[key].get("mape") if key in manifest else None
    except Exception:
        test_mape = result.get("train_mape")

    return ForecastResponse(
        category=result["category"],
        channel=result["channel"],
        model_type=result["model_type"],
        model_spec=result["model_spec"],
        model_class="econometric",
        exog_cols=result["exog_cols"],
        test_mape=test_mape,
        horizon=request.horizon,
        forecast=[ForecastPoint(**p) for p in result["forecast"]])


@router.post("/forecast/upload", response_model=ForecastResponse, tags=["Forecast"])
async def forecast_with_file(category:str,channel:str,horizon:int=12,model_class:str = "econometric",file: UploadFile = File(None)):
    exog_df = None
    if file is not None:
        content = await file.read()
        try:
            if file.filename.endswith((".xlsx", ".xls")):
                exog_df = pd.read_excel(BytesIO(content))
            else:
                _sample = content[:2048].decode("utf-8", errors="ignore")
                _sep = ";" if _sample.count(";") > _sample.count(",") else ","
                exog_df = pd.read_csv(BytesIO(content), sep=_sep)
        except Exception as e:
            raise HTTPException(status_code=422, detail=f"Ошибка чтения файла: {e}")

    if model_class == "ml_v2":
        user_exog = None
        if exog_df is not None:
            user_exog = {col: exog_df[col].tolist() for col in exog_df.columns}
        try:
            from src.forecasting.ml_v2_forecast import forecast_ml_v2
            result = forecast_ml_v2(
                series_dict=_series_dict,
                category=category,
                channel=channel,
                horizon=horizon,
                user_exog=user_exog)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

        return ForecastResponse(
            category=result["category"],
            channel=result["channel"],
            model_type=result["model_type"],
            model_spec=result["model_spec"],
            model_class="ml_v2",
            exog_cols=result["exog_cols"],
            test_mape=result.get("test_mape"),
            horizon=horizon,
            forecast=[ForecastPoint(**p) for p in result["forecast"]])

    try:
        result = forecast_from_manifest(
            series_dict=_series_dict,
            category=category,
            channel=channel,
            horizon=horizon,
            exog_df=exog_df)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка прогноза: {e}")

    try:
        manifest = load_manifest()
        key = f"{category}|{channel}"
        test_mape = manifest[key].get("mape") if key in manifest else None
    except Exception:
        test_mape = result.get("train_mape")

    return ForecastResponse(
        category=result["category"],
        channel=result["channel"],
        model_type=result["model_type"],
        model_spec=result["model_spec"],
        model_class="econometric",
        exog_cols=result["exog_cols"],
        test_mape=test_mape,
        horizon=horizon,
        forecast=[ForecastPoint(**p) for p in result["forecast"]])


@router.post("/forecast/aggregate", response_model=AggregateResponse, tags=["Forecast"])
async def forecast_aggregate(request: AggregateRequest):
    pairs = request.pairs
    if not pairs:
        raise HTTPException(status_code=422, detail="Нет рядов для агрегации")

    def _get_best_model_for_series(cat: str, ch: str) -> str:
        results_dir = ML_RESULTS_DIR.parent / "part3"
        cross_path = results_dir / "cross_model_comparison_per_series.csv"

        if cross_path.exists():
            try:
                df = pd.read_csv(cross_path)
                row = df[(df["category"]==cat)&(df["channel"]==ch)]
                if not row.empty:
                    r = row.iloc[0]
                    candidates = {}
                    for col, mc in [("best_eco_mape", "econometric"),("best_ml_mape","ml_v2"),("lstm_mape","lstm"),]:
                        val = r.get(col)
                        if val is not None and pd.notna(val) and float(val) > 0:
                            candidates[mc] = float(val)
                    if candidates:
                        return min(candidates, key=candidates.get)
            except Exception:
                pass
        return "econometric"

    individual_forecasts = []
    errors = []

    for pair in pairs:
        cat = pair.category if hasattr(pair, "category") else pair["category"]
        ch = pair.channel if hasattr(pair, "channel") else pair["channel"]
        use_model = (_get_best_model_for_series(cat, ch) if request.model_class in ("best", "auto", "") else request.model_class)

        try:
            if use_model == "ml_v2":
                from src.forecasting.ml_v2_forecast import forecast_ml_v2
                fc = forecast_ml_v2(
                    series_dict=_series_dict,
                    category=cat,
                    channel=ch,
                    horizon=request.horizon,
                    user_exog=None)
            elif use_model == "lstm":
                from src.forecasting.lstm_forecast import forecast_lstm, PROD_VARIANT
                fc = await asyncio.to_thread(forecast_lstm,_series_dict_ml,cat,ch,request.horizon,PROD_VARIANT,None)
            else:
                fc = forecast_from_manifest(
                    series_dict=_series_dict,
                    category=cat,
                    channel=ch,
                    horizon=request.horizon)
                try:
                    manifest = load_manifest()
                    key_str = f"{cat}|{ch}"
                    fc["test_mape"] = (manifest[key_str].get("mape") if key_str in manifest else None)
                except Exception:
                    pass

            try:
                from src.forecasting.comparison import get_full_comparison_all_models
                _cmp = get_full_comparison_all_models(cat, ch)
                _winner_model = _cmp.get("winner_model") or fc.get("model_type", use_model)
            except Exception:
                _winner_model = fc.get("model_type", use_model)

            individual_forecasts.append({
                "category":     cat,
                "channel":      ch,
                "model_class":  use_model,
                "model_type":   fc.get("model_type", use_model),
                "winner_model": _winner_model,
                "test_mape":    fc.get("test_mape"),
                "forecast":     fc["forecast"]})
        except Exception as e:
            errors.append({"category": cat, "channel": ch,"model": use_model, "error": str(e)})

    if not individual_forecasts:
        raise HTTPException(status_code=500,detail=f"Не удалось построить ни одного прогноза. Ошибки: {errors}")

    all_points = []
    for fc_item in individual_forecasts:
        all_points.extend(fc_item["forecast"])

    agg_df = (pd.DataFrame(all_points)
        .groupby("date")
        .agg(forecast=("forecast", "sum"),
            lower_80=("lower_80", "sum"),
            upper_80=("upper_80", "sum"))
        .reset_index()
        .sort_values("date"))

    agg_points = [
        ForecastPoint(
            date=row["date"],
            forecast=round(float(row["forecast"]), 2),
            lower_80=round(float(row["lower_80"]), 2),
            upper_80=round(float(row["upper_80"]), 2))
        for _, row in agg_df.iterrows()]

    return AggregateResponse(
        label=request.label or f"Агрегация ({len(individual_forecasts)} рядов)",
        model_class=request.model_class,
        n_series=len(individual_forecasts),
        n_errors=len(errors),
        errors=errors if errors else None,
        forecast=agg_points,
        individual=individual_forecasts)


@router.get("/models/info/{category}/{channel}",response_model=ModelInfo, tags=["Models"])
def get_model_info(category: str, channel: str):
    try:
        manifest = load_manifest()
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    key = f"{category}|{channel}"
    if key not in manifest:
        raise HTTPException(status_code=404, detail=f"Ряд '{key}' не найден в манифесте")

    info = manifest[key]
    return ModelInfo(
        category=info["category"],
        channel=info["channel"],
        best_model=info["best_model"],
        order=info["order"],
        seasonal_order=info["seasonal_order"],
        exog_cols=info["exog_cols"],
        mape=info.get("mape"))


@router.get("/models/ml_v2/info/{category}/{channel}", tags=["Models"])
def get_ml_v2_info(category: str, channel: str):
    try:
        from src.forecasting.ml_v2_forecast import get_comparison, _load_metadata
        meta = _load_metadata()
        selected_exog = meta.get("selected_exog", [])
        architecture = meta.get("architecture", "")
        winner_model = meta.get("winner_v2_model", "XGBoost")
        comp = get_comparison(category, channel)

        return {
            "category":      category,
            "channel":       channel,
            "selected_exog": selected_exog,
            "architecture":  architecture,
            "winner_model":  winner_model,
            "test_mape_no_exog":   comp.get("ml_v2_no_exog"),
            "test_mape_full_exog": comp.get("ml_v2_full_exog")}
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models/comparison/{category}/{channel}",response_model=ModelComparisonInfo, tags=["Models"])
def get_model_comparison(category: str, channel: str):
    try:
        from src.forecasting.ml_v2_forecast import get_comparison
        ml_result = get_comparison(category, channel)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    best_sarima_model = None
    try:
        manifest = load_manifest()
        key = f"{category}|{channel}"
        if key in manifest:
            best_sarima_model = manifest[key].get("best_model")
    except Exception:
        pass

    return ModelComparisonInfo(
        category=category,
        channel=channel,
        sarima_mape=ml_result.get("sarima_mape"),
        ml_v2_no_exog=ml_result.get("ml_v2_no_exog"),
        ml_v2_partial=ml_result.get("ml_v2_partial_exog"),
        ml_v2_full=ml_result.get("ml_v2_full_exog"),
        best_model=ml_result.get("best_model"),
        best_mape=ml_result.get("best_mape"),
        recommendation=ml_result.get("recommendation", ""),
        selected_exog=ml_result.get("selected_exog", []),
        config_name=ml_result.get("config_name", "vif_exog"),
        winner_v2_model=ml_result.get("winner_v2_model", ""),
        best_sarima_model=best_sarima_model or "")


@router.get("/models/all", tags=["Models"])
def get_all_models():
    try:
        manifest = load_manifest()
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    return [
        {
            "category":   v["category"],
            "channel":    v["channel"],
            "best_model": v["best_model"],
            "mape":       v.get("mape"),
            "exog_cols":  v["exog_cols"]}
        for v in manifest.values()]


@router.get("/models/lstm/info", tags=["Models"])
def get_lstm_info_endpoint(variant: str = None):
    try:
        from src.forecasting.lstm_forecast import get_lstm_info, PROD_VARIANT
        v = variant or PROD_VARIANT
        return get_lstm_info(v)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models/lstm/variants", tags=["Models"])
def get_lstm_variants():
    try:
        from src.forecasting.lstm_forecast import list_available_variants
        return {"available_variants": list_available_variants()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models/lstm/comparison/{category}/{channel}", tags=["Models"])
def get_lstm_comparison_endpoint(category: str, channel: str):
    try:
        from src.forecasting.lstm_forecast import get_lstm_comparison
        return get_lstm_comparison(category, channel)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models/full_comparison/{category}/{channel}", tags=["Models"])
def get_full_comparison_endpoint(category: str, channel: str):
    try:
        from src.forecasting.comparison import get_full_comparison_all_models
        return get_full_comparison_all_models(category, channel)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/train", response_model=TrainResponse, tags=["Training"])
def train(request: TrainRequest):
    if request.secret_key != TRAIN_SECRET:
        raise HTTPException(status_code=403, detail="Неверный ключ")

    try:
        from src.training.train_sarima import run_training
        manifest = run_training(use_individual_order=request.use_individual_order,test_size=request.test_size)
        global _df, _series_dict
        _df = load_data(str(DATA_PATH))
        _series_dict = create_series_dict(_df)
        return TrainResponse(status="ok",models_trained=len(manifest),message=f"Обучено {len(manifest)} моделей. Манифест обновлён.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка обучения: {e}")


@router.get("/series/{category}/{channel}/stl",response_model=StlResponse, tags=["Analytics"])
def get_stl(category: str, channel: str):
    result = get_stl_decomposition(_series_dict, category, channel)
    if not result:
        raise HTTPException(status_code=404, detail=f"Ряд '{category}|{channel}' не найден")
    if "error" in result:
        raise HTTPException(status_code=422, detail=result["error"])
    return result


@router.get("/series/{category}/{channel}/stationarity",response_model=StationarityResponse, tags=["Analytics"])
def get_stationarity_tests(category: str, channel: str):
    result = get_stationarity(_series_dict, category, channel)
    if not result:
        raise HTTPException(status_code=404, detail=f"Ряд '{category}|{channel}' не найден")
    return result


@router.get("/series/{category}/{channel}/acf_pacf",response_model=AcfPacfResponse, tags=["Analytics"])
def get_acf_pacf_values(category: str, channel: str,n_lags:int=36, diff: str = "original"):
    result = get_acf_pacf(_series_dict, category, channel, n_lags=n_lags, diff=diff)
    if not result:
        raise HTTPException(status_code=404, detail=f"Ряд '{category}|{channel}' не найден")
    if "error" in result:
        raise HTTPException(status_code=422, detail=result["error"])
    return result


@router.get("/series/{category}/{channel}/hp_filter", tags=["Analytics"])
def get_hp_filter_endpoint(category: str, channel: str, lamb: int = 1600):
    result = get_hp_filter(_series_dict, category, channel, lamb=lamb)
    if not result:
        raise HTTPException(status_code=404, detail=f"Ряд '{category}|{channel}' не найден")
    if "error" in result:
        raise HTTPException(status_code=422, detail=result["error"])
    return result


@router.get("/series/{category}/{channel}/fft", tags=["Analytics"])
def get_fft_endpoint(category: str, channel: str):
    result = get_fft_analysis(_series_dict, category, channel)
    if not result:
        raise HTTPException(status_code=404, detail=f"Ряд '{category}|{channel}' не найден")
    if "error" in result:
        raise HTTPException(status_code=422, detail=result["error"])
    return result


@router.get("/series/{category}/{channel}/mann_kendall", tags=["Analytics"])
def get_mann_kendall_endpoint(category: str, channel: str):
    result = get_mann_kendall(_series_dict, category, channel)
    if not result:
        raise HTTPException(status_code=404, detail=f"Ряд '{category}|{channel}' не найден")
    if "error" in result:
        raise HTTPException(status_code=422, detail=result["error"])
    return result


@router.get("/series/{category}/{channel}/structural_breaks", tags=["Analytics"])
def get_structural_breaks_endpoint(category: str, channel: str):
    result = get_structural_breaks(_series_dict, category, channel)
    if not result:
        raise HTTPException(status_code=404, detail=f"Ряд '{category}|{channel}' не найден")
    if "error" in result:
        raise HTTPException(status_code=422, detail=result["error"])
    return result


@router.get("/series/{category}/{channel}/seasonal_subseries", tags=["Analytics"])
def get_seasonal_subseries_endpoint(category: str, channel: str):
    result = get_seasonal_subseries(_series_dict, category, channel)
    if not result:
        raise HTTPException(status_code=404, detail=f"Ряд '{category}|{channel}' не найден")
    return result


@router.get("/series/{category}/{channel}/seasonality_test", tags=["Analytics"])
def get_seasonality_test_endpoint(category: str, channel: str):
    result = get_seasonality_test(_series_dict, category, channel)
    if not result:
        raise HTTPException(status_code=404, detail=f"Ряд '{category}|{channel}' не найден")
    if "error" in result:
        raise HTTPException(status_code=422, detail=result["error"])
    return result


@router.get("/series/{category}/{channel}/anomalies", tags=["Analytics"])
def get_anomalies_endpoint(category: str, channel: str,method: str = "stl", threshold: float = 2.5):
    result = get_anomalies(_series_dict, category, channel,method=method, threshold=threshold)
    if not result:
        raise HTTPException(status_code=404, detail=f"Ряд '{category}|{channel}' не найден")
    return result


@router.get("/analytics/covid_impact", tags=["Analytics"])
def get_covid_impact_endpoint(channel: str = None):
    series = _series_dict
    if channel:
        series = {k: v for k,v in _series_dict.items() if k[1] == channel}
    return get_covid_impact(series)


@router.get("/analytics/correlation", tags=["Analytics"])
def get_correlation_endpoint(channel: str = None):
    result = get_cross_series_correlation(_series_dict, filter_channel=channel)
    if "error" in result:
        raise HTTPException(status_code=422, detail=result["error"])
    return result


@router.get("/analytics/cross_correlation", tags=["Analytics"])
def get_cross_correlation_endpoint(category_a: str, channel_a: str,category_b: str, channel_b: str,n_lags: int = 12):
    result = get_cross_correlation(
        _series_dict,
        category_a, channel_a,
        category_b, channel_b,
        n_lags=n_lags)
    if "error" in result:
        raise HTTPException(status_code=422, detail=result["error"])
    return result


@router.get("/analytics/summary", tags=["Analytics"])
def get_analytics_summary():
    return get_full_analytics_summary(_series_dict)