import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mae  = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))

    nonzero = np.abs(y_true) > 1.0
    mape = float(np.mean(np.abs(
        (y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero]
    )) * 100) if nonzero.sum() > 0 else np.nan

    denom = (np.abs(y_true) + np.abs(y_pred)) / 2
    mask  = denom > 1.0
    smape = float(np.mean(
        np.abs(y_true[mask] - y_pred[mask]) / denom[mask]
    ) * 100) if mask.sum() > 0 else np.nan

    return {"mae": mae, "rmse": rmse, "mape": mape, "smape": smape}