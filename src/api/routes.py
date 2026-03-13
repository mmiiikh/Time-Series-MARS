import os
import pandas as pd
from io import BytesIO
from fastapi import APIRouter, HTTPException, UploadFile, File, Depends

from src.api.schemas import (
    SeriesInfo, HistoryResponse,
    StlResponse, StationarityResponse, AcfPacfResponse,
    ForecastRequest, ForecastResponse, ForecastPoint,
    ModelInfo, TrainRequest, TrainResponse,
)
from src.data.load_data import load_data, create_series_dict
from src.data.preprocess import (
    get_series_stats, get_series_history,
    get_stl_decomposition, get_stationarity, get_acf_pacf,
)
from src.forecasting.sarima import forecast_from_manifest, load_manifest
from src.config.settings import DATA_PATH

router = APIRouter()

_df          = load_data(str(DATA_PATH))
_series_dict = create_series_dict(_df)

TRAIN_SECRET = os.getenv("TRAIN_SECRET_KEY", "mars-train-2024")


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
                         n_lags: int = 36,
                         diff: str = "original"):
    result = get_acf_pacf(_series_dict, category, channel, n_lags=n_lags, diff=diff)
    if not result:
        raise HTTPException(status_code=404, detail=f"Ряд '{category}|{channel}' не найден")
    if "error" in result:
        raise HTTPException(status_code=422, detail=result["error"])
    return result

@router.post("/forecast", response_model=ForecastResponse, tags=["Forecast"])
def forecast(request: ForecastRequest):
    if request.model_class not in ("econometric", "ml", "dl"):
        raise HTTPException(status_code=400,
                            detail="model_class должен быть: econometric | ml | dl")

    if request.model_class in ("ml", "dl"):
        raise HTTPException(status_code=501,
                            detail=f"Модели класса '{request.model_class}' ещё не реализованы")

    exog_df = None
    if request.exog_data:
        try:
            exog_df = pd.DataFrame(request.exog_data)
        except Exception as e:
            raise HTTPException(status_code=422,
                                detail=f"Ошибка в exog_data: {e}")

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

    return ForecastResponse(
        category=result["category"],
        channel=result["channel"],
        model_type=result["model_type"],
        model_spec=result["model_spec"],
        model_class=request.model_class,
        exog_cols=result["exog_cols"],
        train_mape=result["train_mape"],
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

    return ForecastResponse(
        category=result["category"],
        channel=result["channel"],
        model_type=result["model_type"],
        model_spec=result["model_spec"],
        model_class=model_class,
        exog_cols=result["exog_cols"],
        train_mape=result["train_mape"],
        horizon=horizon,
        forecast=[ForecastPoint(**p) for p in result["forecast"]],
    )


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
        mape=info.get("mape"),
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