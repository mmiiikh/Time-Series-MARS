import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    prefix: str = "") -> dict:
    """
    MAE, RMSE, MAPE, sMAPE.
    Фильтрует нулевые значения чтобы MAPE не уходил в бесконечность.
    """
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)

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

    return {
        f"{prefix}mae":   round(mae,  0),
        f"{prefix}rmse":  round(rmse, 0),
        f"{prefix}mape":  round(mape,  2) if not np.isnan(mape)  else np.nan,
        f"{prefix}smape": round(smape, 2) if not np.isnan(smape) else np.nan,
    }


def coverage_and_width(y_true: np.ndarray,
                        lower: np.ndarray,
                        upper: np.ndarray) -> dict:
    """Покрытие и ширина доверительного интервала."""
    coverage  = float(np.mean((y_true >= lower) & (y_true <= upper)))
    avg_width = float(np.mean(upper - lower))
    return {
        "coverage":  round(coverage,  3),
        "avg_width": round(avg_width, 0),
    }