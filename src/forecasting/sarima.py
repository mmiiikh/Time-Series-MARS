import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX


class SarimaModel:
    def __init__(self, order, seasonal_order):
        self.order = order
        self.seasonal_order = seasonal_order
        self.result = None

    def fit(self, series):
        model = SARIMAX(
            np.log(series),
            order=self.order,
            seasonal_order=self.seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False)
        self.result = model.fit(disp=False)

    def forecast(self, steps: int):
        forecast_log = self.result.get_forecast(steps=steps).predicted_mean
        return np.exp(forecast_log)