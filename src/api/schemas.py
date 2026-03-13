from pydantic import BaseModel, Field
from typing import Optional

class SeriesInfo(BaseModel):
    category:  str
    channel:   str
    n_obs:     int
    date_from: str
    date_to:   str


class HistoryResponse(BaseModel):
    category: str
    channel:  str
    dates:    list[str]
    values:   list[float]

class StlMetrics(BaseModel):
    Fs_seasonal:    float
    Ft_trend:       float
    R2:             float
    seas_amplitude: float
    trend_slope_mo: float


class StlResponse(BaseModel):
    category: str
    channel:  str
    dates:    list[str]
    observed: list[float]
    trend:    list[float]
    seasonal: list[float]
    resid:    list[float]
    metrics:  StlMetrics


class StationarityResult(BaseModel):
    adf_stat:      float
    adf_p:         float
    adf_1pct:      float
    kpss_stat:     float
    kpss_p:        float
    conclusion:    str
    is_stationary: bool


class StationarityResponse(BaseModel):
    category: str
    channel:  str
    tests: dict[str, StationarityResult]


class AcfPacfResponse(BaseModel):
    category:      str
    channel:       str
    diff:          str
    lags:          list[int]
    acf:           list[float]
    pacf:          list[float]
    conf_bound:    float
    seasonal_lags: list[int]


class ForecastRequest(BaseModel):
    category:   str
    channel:    str
    # "econometric" | "ml" | "dl"  (ml и dl появятся в частях 2 и 3)
    model_class: str = Field(default="econometric",
                             description="Класс модели: econometric | ml | dl")
    horizon:    int  = Field(default=12, ge=1, le=60)
    exog_data:  Optional[dict[str, list[float]]] = None


class ForecastPoint(BaseModel):
    date:      str
    forecast:  float
    lower_80:  float
    upper_80:  float


class ForecastResponse(BaseModel):
    category:   str
    channel:    str
    model_type: str
    model_spec: str
    model_class: str
    exog_cols:  list[str]
    train_mape: Optional[float]
    horizon:    int
    forecast:   list[ForecastPoint]

class ModelInfo(BaseModel):
    category:      str
    channel:       str
    best_model:    str
    order:         list[int]
    seasonal_order: list[int]
    exog_cols:     list[str]
    mape:          Optional[float]

class TrainRequest(BaseModel):
    use_individual_order: bool = True
    test_size:            int  = 12
    secret_key:           str  = Field(..., description="Ключ для защиты эндпоинта")


class TrainResponse(BaseModel):
    status:        str
    models_trained: int
    message:       str