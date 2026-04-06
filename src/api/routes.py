"""
src/api/routes.py — FastAPI роуты Mars Forecasting Service.

Изменения v2:
  - ForecastResponse: train_mape → test_mape (MAPE на тестовом периоде)
  - /forecast поддерживает model_class="ml_v2"
  - GET /models/comparison/{category}/{channel} — сравнение SARIMA vs ML v2
  - GET /models/best/{category}/{channel}        — рекомендация лучшей модели
"""

import os
import pandas as pd
from io import BytesIO
from fastapi import APIRouter, HTTPException, UploadFile, File

from src.api.schemas import (
    SeriesInfo, HistoryResponse,
    StlResponse, StationarityResponse, AcfPacfResponse,
    ForecastRequest, ForecastResponse, ForecastPoint,
    ModelInfo, ModelComparisonInfo,
    TrainRequest, TrainResponse,
)
from src.data.load_data import load_data, create_series_dict
from src.data.preprocess import (
    get_series_stats, get_series_history,
    get_stl_decomposition, get_stationarity, get_acf_pacf,
)
from src.forecasting.sarima import forecast_from_manifest, load_manifest
from src.config.settings import DATA_PATH

from src.data.analytics import (
    get_hp_filter,
    get_fft_analysis,
    get_mann_kendall,
    get_structural_breaks,
    get_seasonal_subseries,
    get_seasonality_test,
    get_anomalies,
    get_covid_impact,
    get_cross_series_correlation,
    get_cross_correlation,
    get_full_analytics_summary,
)

router = APIRouter()

_df          = load_data(str(DATA_PATH))
_series_dict = create_series_dict(_df)

TRAIN_SECRET = os.getenv("TRAIN_SECRET_KEY", "mars-train-2024")


# =============================================================================
# Series
# =============================================================================

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


# =============================================================================
# Forecast
# =============================================================================

@router.post("/forecast", response_model=ForecastResponse, tags=["Forecast"])
def forecast(request: ForecastRequest):
    if request.model_class not in ("econometric", "ml_v2", "dl"):
        raise HTTPException(
            status_code=400,
            detail="model_class должен быть: econometric | ml_v2",
        )

    if request.model_class == "dl":
        raise HTTPException(status_code=501, detail="DL модели ещё не реализованы")

    # ── ML v2 ────────────────────────────────────────────────────────────────
    if request.model_class == "ml_v2":
        try:
            from src.forecasting.ml_v2_forecast import forecast_ml_v2
            result = forecast_ml_v2(
                series_dict=_series_dict,
                category=request.category,
                channel=request.channel,
                horizon=request.horizon,
                user_exog=request.exog_data,
            )
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Ошибка ML v2 прогноза: {e}")

        return ForecastResponse(
            category=result["category"],
            channel=result["channel"],
            model_type=result["model_type"],
            model_spec=result["model_spec"],
            model_class="ml_v2",
            exog_cols=result["exog_cols"],
            test_mape=result.get("test_mape"),
            horizon=request.horizon,
            forecast=[ForecastPoint(**p) for p in result["forecast"]],
        )

    # ── Econometric (SARIMA/SARIMAX) ─────────────────────────────────────────
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
            exog_df=exog_df,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка прогноза: {e}")

    # test_mape берём из манифеста (MAPE на тестовом периоде, не train)
    try:
        manifest  = load_manifest()
        key       = f"{request.category}|{request.channel}"
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
        forecast=[ForecastPoint(**p) for p in result["forecast"]],
    )


@router.post("/forecast/upload", response_model=ForecastResponse, tags=["Forecast"])
async def forecast_with_file(
    category:    str,
    channel:     str,
    horizon:     int = 12,
    model_class: str = "econometric",
    file: UploadFile = File(None),
):
    exog_df = None
    if file is not None:
        content = await file.read()
        try:
            if file.filename.endswith((".xlsx", ".xls")):
                exog_df = pd.read_excel(BytesIO(content))
            else:
                exog_df = pd.read_csv(BytesIO(content))
        except Exception as e:
            raise HTTPException(status_code=422, detail=f"Ошибка чтения файла: {e}")

    if model_class == "ml_v2":
        user_exog = None
        if exog_df is not None:
            user_exog = {col: exog_df[col].tolist()
                         for col in exog_df.columns if col in exog_df.columns}
        try:
            from src.forecasting.ml_v2_forecast import forecast_ml_v2
            result = forecast_ml_v2(
                series_dict=_series_dict,
                category=category,
                channel=channel,
                horizon=horizon,
                user_exog=user_exog,
            )
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
            forecast=[ForecastPoint(**p) for p in result["forecast"]],
        )

    try:
        result = forecast_from_manifest(
            series_dict=_series_dict,
            category=category,
            channel=channel,
            horizon=horizon,
            exog_df=exog_df,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка прогноза: {e}")

    try:
        manifest  = load_manifest()
        key       = f"{category}|{channel}"
        test_mape = manifest[key].get("mape") if key in manifest else None
    except Exception:
        test_mape = result.get("train_mape")

    return ForecastResponse(
        category=result["category"],
        channel=result["channel"],
        model_type=result["model_type"],
        model_spec=result["model_spec"],
        model_class=model_class,
        exog_cols=result["exog_cols"],
        test_mape=test_mape,
        horizon=horizon,
        forecast=[ForecastPoint(**p) for p in result["forecast"]],
    )


# =============================================================================
# Models
# =============================================================================

@router.get("/models/info/{category}/{channel}",
            response_model=ModelInfo, tags=["Models"])
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
        mape=info.get("mape"),   # MAPE на тестовом периоде (из манифеста)
    )


@router.get("/models/comparison/{category}/{channel}",
            response_model=ModelComparisonInfo, tags=["Models"])
def get_model_comparison(category: str, channel: str):
    """
    Сравнение SARIMA vs ML v2 для конкретного ряда.
    Оба MAPE считаются по одному протоколу: h=1..12 из одной точки T на тестовых 12 месяцах.
    Используется для рекомендации лучшей модели в UI.
    """
    try:
        from src.forecasting.ml_v2_forecast import get_comparison
        result = get_comparison(category, channel)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return ModelComparisonInfo(
        category=category,
        channel=channel,
        sarima_mape=result.get("sarima_mape"),
        ml_v2_no_exog=result.get("ml_v2_no_exog"),
        ml_v2_partial=result.get("ml_v2_partial_exog"),
        ml_v2_full=result.get("ml_v2_full_exog"),
        best_model=result.get("best_model"),
        best_mape=result.get("best_mape"),
        recommendation=result.get("recommendation", ""),
        selected_exog=result.get("selected_exog", []),
        config_name=result.get("config_name", "vif_exog"),
        winner_v2_model=result.get("winner_v2_model", ""),
    )


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
            "exog_cols":  v["exog_cols"],
        }
        for v in manifest.values()
    ]


# =============================================================================
# Training
# =============================================================================

@router.post("/train", response_model=TrainResponse, tags=["Training"])
def train(request: TrainRequest):
    if request.secret_key != TRAIN_SECRET:
        raise HTTPException(status_code=403, detail="Неверный ключ")

    try:
        from src.training.train_sarima import run_training
        manifest = run_training(
            use_individual_order=request.use_individual_order,
            test_size=request.test_size,
        )
        return TrainResponse(
            status="ok",
            models_trained=len(manifest),
            message=f"Обучено {len(manifest)} моделей. Манифест обновлён.",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка обучения: {e}")


# =============================================================================
# Analytics
# =============================================================================

@router.get("/series/{category}/{channel}/stl",
            response_model=StlResponse, tags=["Analytics"])
def get_stl(category: str, channel: str):
    result = get_stl_decomposition(_series_dict, category, channel)
    if not result:
        raise HTTPException(status_code=404, detail=f"Ряд '{category}|{channel}' не найден")
    if "error" in result:
        raise HTTPException(status_code=422, detail=result["error"])
    return result


@router.get("/series/{category}/{channel}/stationarity",
            response_model=StationarityResponse, tags=["Analytics"])
def get_stationarity_tests(category: str, channel: str):
    result = get_stationarity(_series_dict, category, channel)
    if not result:
        raise HTTPException(status_code=404, detail=f"Ряд '{category}|{channel}' не найден")
    return result


@router.get("/series/{category}/{channel}/acf_pacf",
            response_model=AcfPacfResponse, tags=["Analytics"])
def get_acf_pacf_values(category: str, channel: str,
                         n_lags: int = 36, diff: str = "original"):
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
def get_anomalies_endpoint(category: str, channel: str,
                            method: str = "stl", threshold: float = 2.5):
    result = get_anomalies(_series_dict, category, channel,
                           method=method, threshold=threshold)
    if not result:
        raise HTTPException(status_code=404, detail=f"Ряд '{category}|{channel}' не найден")
    return result


@router.get("/analytics/covid_impact", tags=["Analytics"])
def get_covid_impact_endpoint(channel: str = None):
    series = _series_dict
    if channel:
        series = {k: v for k, v in _series_dict.items() if k[1] == channel}
    return get_covid_impact(series)


@router.get("/analytics/correlation", tags=["Analytics"])
def get_correlation_endpoint(channel: str = None):
    result = get_cross_series_correlation(_series_dict, filter_channel=channel)
    if "error" in result:
        raise HTTPException(status_code=422, detail=result["error"])
    return result


@router.get("/analytics/cross_correlation", tags=["Analytics"])
def get_cross_correlation_endpoint(
        category_a: str, channel_a: str,
        category_b: str, channel_b: str,
        n_lags: int = 12,
):
    result = get_cross_correlation(
        _series_dict,
        category_a, channel_a,
        category_b, channel_b,
        n_lags=n_lags,
    )
    if "error" in result:
        raise HTTPException(status_code=422, detail=result["error"])
    return result


@router.get("/analytics/summary", tags=["Analytics"])
def get_analytics_summary():
    return get_full_analytics_summary(_series_dict)