from __future__ import annotations
import warnings
warnings.filterwarnings("ignore")
import json
import numpy as np
import pandas as pd
from pathlib import Path
import torch
import mlflow
import mlflow.pytorch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger
try:
    from pytorch_forecasting import TimeSeriesDataSet, DeepAR
    from pytorch_forecasting.data import GroupNormalizer
    from pytorch_forecasting.metrics import NormalDistributionLoss
except ImportError:
    raise ImportError(
        "Установите зависимости:\n"
        "    pip install pytorch-forecasting lightning\n"
    )

from src.config.settings import (
    TARGET_COL, RANDOM_STATE, HORIZON, SEASONAL_PERIOD,
    MLFLOW_TRACKING_URI, ML_MODELS_DIR, ML_RESULTS_DIR)
from src.data.load_data import load_data, create_series_dict
from src.utils.metrics import compute_metrics


RESULTS_DIR= ML_RESULTS_DIR.parent / "part3"
MODELS_DIR = ML_MODELS_DIR.parent / "dl"
MLFLOW_EXPERIMENT= "mars_deepar"
MLFLOW_EXPERIMENT_PS = "mars_deepar_per_series"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

DEEPAR_CONFIG = {
    "max_encoder_length": 2*SEASONAL_PERIOD,
    "max_prediction_length": HORIZON,
    "hidden_size": 64,
    "rnn_layers":2,
    "dropout": 0.2,
    "learning_rate":1e-3,
    "max_epochs":100,
    "batch_size": 64,
    "gradient_clip_val": 1.0,
    "patience":10,
    "n_samples":100}

pl.seed_everything(RANDOM_STATE, workers=True)


def prepare_dataframe(series_dict: dict) -> pd.DataFrame:
    rows = []
    for (cat, ch), df_s in series_dict.items():
        uid = f"{cat}|{ch}"
        y   = df_s[TARGET_COL].dropna()
        for date, val in y.items():
            rows.append({"unique_id":uid,"ds":  pd.Timestamp(date),"y": float(val)})

    df = pd.DataFrame(rows).sort_values(["unique_id", "ds"]).reset_index(drop=True)
    date_to_idx = {d: i for i, d in enumerate(sorted(df["ds"].unique()))}
    df["time_idx"] = df["ds"].map(date_to_idx)
    return df


def build_datasets(df: pd.DataFrame) -> tuple:
    max_time = df["time_idx"].max()
    cutoff   = max_time-HORIZON+1
    train_df = df[df["time_idx"] < cutoff].copy()
    training = TimeSeriesDataSet(
        train_df,
        time_idx = "time_idx",
        target = "y",
        group_ids= ["unique_id"],
        min_encoder_length = DEEPAR_CONFIG["max_encoder_length"],
        max_encoder_length = DEEPAR_CONFIG["max_encoder_length"],
        min_prediction_length = DEEPAR_CONFIG["max_prediction_length"],
        max_prediction_length = DEEPAR_CONFIG["max_prediction_length"],
        target_normalizer = GroupNormalizer(groups = ["unique_id"],transformation = "softplus"),
        time_varying_known_reals = ["time_idx"],
        time_varying_unknown_reals = ["y"],
        add_relative_time_idx = True,
        add_target_scales= True,
        add_encoder_length = True,
        allow_missing_timesteps = True)

    validation = TimeSeriesDataSet.from_dataset(training,df, predict=True,stop_randomization=True)
    return training, validation

def train_model(
    training:   TimeSeriesDataSet,
    validation: TimeSeriesDataSet) -> tuple[DeepAR, pl.Trainer]:

    train_loader = training.to_dataloader(train = True,batch_size = DEEPAR_CONFIG["batch_size"],num_workers = 0)
    val_loader = validation.to_dataloader(train= False,batch_size = DEEPAR_CONFIG["batch_size"],num_workers = 0)

    model = DeepAR.from_dataset(
        training,
        learning_rate  = DEEPAR_CONFIG["learning_rate"],
        hidden_size = DEEPAR_CONFIG["hidden_size"],
        rnn_layers = DEEPAR_CONFIG["rnn_layers"],
        dropout = DEEPAR_CONFIG["dropout"],
        loss = NormalDistributionLoss(),
        log_interval = 10,
        log_val_interval = 1)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Параметров: {total_params:,}")


    early_stop = EarlyStopping(monitor = "val_loss",patience = DEEPAR_CONFIG["patience"],mode= "min")
    checkpoint_cb = ModelCheckpoint(
        dirpath = str(MODELS_DIR),
        filename = "deepar_best",
        monitor = "val_loss",
        mode = "min",
        save_top_k = 1)

    mlf_logger = MLFlowLogger(experiment_name = MLFLOW_EXPERIMENT,tracking_uri = MLFLOW_TRACKING_URI,
        run_name = "deepar_training",
        log_model = False)

    trainer = pl.Trainer(
        max_epochs = DEEPAR_CONFIG["max_epochs"],
        gradient_clip_val  = DEEPAR_CONFIG["gradient_clip_val"],
        callbacks = [early_stop, checkpoint_cb],
        logger = mlf_logger,
        enable_progress_bar = True,
        enable_model_summary = False,
        deterministic = True)
    trainer.fit(model, train_loader, val_loader)
    best_path = checkpoint_cb.best_model_path
    if best_path:
        model = DeepAR.load_from_checkpoint(best_path)
        print(f"Лучший checkpoint: {best_path}")
    return model, trainer


def predict_and_evaluate(
    model: DeepAR,
    validation: TimeSeriesDataSet,
    df_full: pd.DataFrame,
    df_test: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    val_loader = validation.to_dataloader(
        train = False,
        batch_size = DEEPAR_CONFIG["batch_size"],
        num_workers = 0)

    raw_predictions = model.predict(val_loader,mode = "raw",return_index = True,trainer_kwargs = {"accelerator": "auto"})

    output = raw_predictions.output
    index_df = raw_predictions.index
    def _extract_tensor(output):
        if hasattr(output, "prediction"):
            t = output.prediction
        elif hasattr(output, "samples"):
            t = output.samples
        elif isinstance(output, torch.Tensor):
            t = output
        elif isinstance(output, (list, tuple)) and len(output) > 0:
            t = output[0]
        else:
            raise ValueError(f"Неизвестный формат Output: {type(output)}, attrs: {dir(output)}")
        return t.cpu().numpy() if hasattr(t, "cpu") else np.array(t)
    preds_np = _extract_tensor(output)
    print(f"  Output shape: {preds_np.shape}  dtype: {preds_np.dtype}")
    fc_rows = []
    for i, row in index_df.iterrows():
        uid= row["unique_id"]
        t_start = int(row["time_idx"])
        if preds_np.ndim == 3:
            p_i = preds_np[i]
            if p_i.shape[0] > p_i.shape[-1]:
                lo = np.quantile(p_i, 0.10, axis=0)
                med = np.quantile(p_i, 0.50, axis=0)
                hi = np.quantile(p_i, 0.90, axis=0)
            else:
                lo = p_i[:, 0]
                med = p_i[:, 1] if p_i.shape[1] > 2 else p_i[:, 0]
                hi = p_i[:, -1]
        elif preds_np.ndim == 2:
            med = preds_np[i]
            lo = med*0.85
            hi = med*1.15
        else:
            continue

        for h in range(min(DEEPAR_CONFIG["max_prediction_length"], len(med))):
            fc_rows.append({
                "unique_id":uid,
                "time_idx": t_start + h,
                "forecast_lo80": float(lo[h]),
                "forecast": float(med[h]),
                "forecast_hi80": float(hi[h])})

    fc_df = pd.DataFrame(fc_rows)
    merged = fc_df.merge(df_full[["unique_id", "time_idx", "y", "ds"]],on = ["unique_id", "time_idx"],how = "inner")

    per_series = {}
    for uid, grp in merged.groupby("unique_id"):
        cat, ch = uid.split("|", 1)
        y_true = grp["y"].values.astype(float)
        y_pred = np.maximum(grp["forecast"].values.astype(float), 0)
        m = compute_metrics(y_true, y_pred)
        per_series[(cat, ch)] = {
            "no_exog": m,
            "y_pred": y_pred.tolist(),
            "y_lo":grp["forecast_lo80"].values.tolist(),
            "y_hi":grp["forecast_hi80"].values.tolist()}
    mapes = [r["no_exog"]["mape"] for r in per_series.values()]
    summary = {"mape_no_exog_median": round(float(np.median(mapes)), 2) if mapes else None,
        "mape_no_exog_mean":   round(float(np.mean(mapes)),   2) if mapes else None}
    return merged, {"per_series": per_series, "summary": summary}


def log_per_series_to_mlflow(eval_result: dict) -> None:
    per_series = eval_result.get("per_series", {})
    if not per_series:
        return
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT_PS)
    except Exception as e:
        print(f"MLflow: {e}")
        return

    logged = 0
    mape_vals  = []
    try:
        for (cat, ch), res in per_series.items():
            with mlflow.start_run(run_name=f"{cat}|{ch}"):
                mlflow.set_tag("category",cat)
                mlflow.set_tag("channel",ch)
                mlflow.set_tag("model_class","deepar")
                mlflow.set_tag("library","pytorch_forecasting")
                m = res.get("no_exog", {})
                metrics = {}
                if m.get("mape") is not None and not np.isnan(float(m["mape"])):
                    metrics["deepar_mape_no_exog"] = round(float(m["mape"]), 3)
                    mape_vals.append(float(m["mape"]))
                if m.get("mae") is not None:
                    metrics["deepar_mae_no_exog"] = round(float(m["mae"]), 1)
                if metrics:
                    mlflow.log_metrics(metrics)
            logged+=1

        with mlflow.start_run(run_name="_summary_deepar"):
            mlflow.set_tag("run_type", "summary")
            sm = {"total_series": float(len(per_series))}
            if mape_vals:
                sm["median_mape_no_exog"] = round(float(np.median(mape_vals)),3)
                sm["mean_mape_no_exog"] = round(float(np.mean(mape_vals)),3)
            mlflow.log_metrics(sm)
        if mape_vals:
            print(f"Медиана MAPE: {np.median(mape_vals):.2f}%")
    except Exception as e:
        print(f"MLflow per-series: {e}")


def compare_all_models(eval_result: dict) -> pd.DataFrame:
    rows = []
    for (cat, ch), res in eval_result.get("per_series", {}).items():
        mape = res.get("no_exog", {}).get("mape")
        rows.append({"category":cat,"channel":ch,"deepar_mape": round(float(mape), 2) if mape is not None else None})
    df_dar = pd.DataFrame(rows)
    df_dar.to_csv(RESULTS_DIR / "deepar_per_series.csv", index=False)

    cross_path = RESULTS_DIR / "cross_model_comparison_per_series.csv"
    if cross_path.exists():
        df = pd.read_csv(cross_path)
        df = df[[c for c in df.columns if c != "deepar_mape"]]
        df = df.merge(df_dar, on=["category", "channel"], how="outer")
    else:
        print("cross_model_comparison_per_series.csv не найден")
        df = df_dar

    label_map = {"best_eco_mape": "Эконометрика",
        "best_ml_mape":"ML",
        "lstm_mape": "LSTM",
        "deepar_mape":"DeepAR"}
    mape_cols = [c for c in label_map if c in df.columns]

    def best_row(row):
        vals = {label_map[c]: row[c] for c in mape_cols if pd.notna(row.get(c))}
        return min(vals, key=vals.get) if vals else "-"
    df["Победитель"] = df.apply(best_row, axis=1)

    for c in mape_cols:
        med = df[c].dropna().median()
        label = label_map[c]
    print(f"Победитель по рядам:")
    for lbl, cnt in df["Победитель"].value_counts().items():
        print(f"{lbl}: {cnt} рядов")

    out = RESULTS_DIR / "cross_model_comparison_with_deepar.csv"
    df.to_csv(out, index=False)
    return df

def main() -> dict:

    from src.config.settings import ML_DATA_FILE
    df_raw = load_data(str(ML_DATA_FILE))
    series_dict = create_series_dict(df_raw)
    df_full = prepare_dataframe(series_dict)
    max_time = df_full["time_idx"].max()
    df_test  = df_full[df_full["time_idx"] > max_time-HORIZON].copy()
    training, validation = build_datasets(df_full)
    model, trainer = train_model(training, validation)
    best_val = trainer.callback_metrics.get("val_loss", float("nan"))
    fc_df, eval_result = predict_and_evaluate(model, validation, df_full, df_test)
    summary = eval_result["summary"]
    print(f"Медиана MAPE: {summary.get('mape_no_exog_median')}%")
    print(f"Средняя MAPE: {summary.get('mape_no_exog_mean')}%")
    fc_path = RESULTS_DIR / "deepar_forecasts.csv"
    fc_df.to_csv(fc_path, index=False)
    print(f"Прогнозы: {fc_path}")
    log_per_series_to_mlflow(eval_result)
    compare_all_models(eval_result)
    return {"model": model,"trainer": trainer,"eval_result": eval_result,"fc_df": fc_df}


if __name__ == "__main__":
    main()