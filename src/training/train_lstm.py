from __future__ import annotations
import json
import warnings
import math
from pathlib import Path
import numpy as np
import pandas as pd
import mlflow
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.config.settings import (
    ML_DATA_FILE, ML_MODELS_DIR, MLFLOW_TRACKING_URI,
    TARGET_COL, EXOG_COLS, RANDOM_STATE, TEST_SIZE,
    DL_DIR, ML_RESULTS_DIR)
from src.data.load_data import load_data, create_series_dict
from src.data.preprocess import build_global_dataset
from src.utils.metrics import compute_metrics
from src.forecasting.lstm_model import (
    LSTMForecaster, LSTMAttentionForecaster,
    TimeSeriesDataset, SequenceScaler, EarlyStopping,
    build_model_from_config,
    VIF_EXOG_VARS, BASE_FEATURE_GROUPS, FUTURE_EXOG_COLS,
    WINDOW_SIZE, HORIZON, safe_col,
    get_base_feature_names, get_exog_feature_names)

warnings.filterwarnings("ignore")
torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)


MLFLOW_EXPERIMENT_LSTM = "mars_lstm"
RESULTS3 = ML_RESULTS_DIR.parent / "part3"
DL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS3.mkdir(parents=True, exist_ok=True)
LSTM_CONFIG = {"embed_dim":4,"hidden_size": 64,"n_layers":2,"dropout":0.25,"horizon":HORIZON}

TRAIN_CONFIG = {"batch_size":32,"lr": 1e-3,"weight_decay": 1e-4,"max_epochs":300,"patience": 25,"grad_clip": 1.0,"val_fraction": 0.15,"window_size":WINDOW_SIZE}

TEST_HORIZONS = list(range(1, HORIZON+1))

DEVICE = (torch.device("mps")  if torch.backends.mps.is_available() else
    torch.device("cuda") if torch.cuda.is_available() else
    torch.device("cpu"))


def step_load_data() -> tuple:
    lag_ranges = {}
    meta_path  = ML_MODELS_DIR / "ml_v2_vif_exog_metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        lag_ranges = meta.get("ccf_lag_ranges", {})
        print(f"lag_ranges из {meta_path.name}")
    else:
        print("ml_v2_vif_exog_metadata.json не найден")
    df_raw = load_data(str(ML_DATA_FILE))
    series_dict = create_series_dict(df_raw)
    global_df, feature_cols, cat_enc, ch_enc = build_global_dataset(series_dict, lag_ranges)

    categories = sorted(global_df["_category"].unique())
    channels = sorted(global_df["_channel"].unique())
    return df_raw, series_dict, global_df, feature_cols, cat_enc, ch_enc, lag_ranges, categories, channels


def step_build_features(
    global_df: pd.DataFrame,
    series_dict: dict,
    categories: list,
    channels: list) -> tuple[pd.DataFrame, dict, dict]:

    df = global_df.copy()
    df["_date"] = pd.to_datetime(df["_date"])
    cat_to_id = {c: i for i, c in enumerate(categories)}
    ch_to_id = {c: i for i, c in enumerate(channels)}
    df["_cat_id"] = df["_category"].map(cat_to_id)
    df["_ch_id"] = df["_channel"].map(ch_to_id)
    if "t" in df.columns and "t_squared" not in df.columns:
        df["t_squared"] = df["t"]**2

    exog_frames = []
    for (cat, ch), raw_df in series_dict.items():
        cols_here = [c for c in VIF_EXOG_VARS if c in raw_df.columns]
        if not cols_here:
            continue
        tmp = raw_df[cols_here].copy()
        tmp["_category"] = cat
        tmp["_channel"] = ch
        tmp["_date"] = tmp.index
        exog_frames.append(tmp.reset_index(drop=True))

    if exog_frames:
        exog_merged = pd.concat(exog_frames, ignore_index=True)
        exog_merged["_date"] = pd.to_datetime(exog_merged["_date"])
        df = df.merge(exog_merged,on=["_category", "_channel", "_date"],how="left",suffixes=("", "_raw")).reset_index(drop=True)

    exog_lag_cols = []
    for col in VIF_EXOG_VARS:
        if col not in df.columns:
            continue
        for lag in [1, 2]:
            lag_name = f"{safe_col(col)}_lag{lag}"
            df[lag_name] = df.groupby(["_category", "_channel"])[col].shift(lag)
            exog_lag_cols.append(lag_name)

    dates = sorted(df["_date"].unique())
    train_cutoff = max(set(dates[:-TEST_SIZE]))
    last_value: dict[tuple, dict[str, float]] = {}
    for (cat, ch), raw_df in series_dict.items():
        train_slice = raw_df[raw_df.index <= pd.Timestamp(train_cutoff)]
        lv = {}
        for col in VIF_EXOG_VARS:
            if col in train_slice.columns:
                s = train_slice[col].dropna()
                lv[col] = float(s.iloc[-1]) if len(s) > 0 else 0.0
            else:
                lv[col] = 0.0
        last_value[(cat, ch)] = lv
    global_last = {col: float(np.median([v[col] for v in last_value.values() if col in v])) for col in VIF_EXOG_VARS}

    for col in VIF_EXOG_VARS:
        if col not in df.columns:
            df[safe_col(col)+"_future"] = 0.0
            continue
        fc = safe_col(col)+"_future"
        df[fc] = df.groupby(["_category", "_channel"])[col].shift(-1)

    for col in VIF_EXOG_VARS:
        sc = safe_col(col)
        for suffix in ["_lag1", "_lag2", "_future"]:
            cname = sc + suffix
            if cname not in df.columns:
                continue
            for (cat, ch), grp_idx in df.groupby(["_category", "_channel"]).groups.items():
                fill_val = last_value.get((cat, ch), {}).get(col, global_last.get(col, 0.0))
                df.loc[grp_idx, cname] = df.loc[grp_idx, cname].fillna(fill_val)

    base_feats = get_base_feature_names()
    missing = [f for f in base_feats if f not in df.columns]
    if missing:
        print(f"Отсутствующие базовые признаки: {missing}")
        for m in missing:
            df[m] = 0.0

    avail_base = [f for f in base_feats  if f in df.columns]
    avail_exog = [f for f in get_exog_feature_names() if f in df.columns]

    print(f"Базовых признаков: {len(avail_base)}  ({len(avail_base)} из {len(base_feats)})")
    print(f"Экзогенных (base + future): {len(avail_exog)}")
    print(f"last_value_per_series: {len(last_value)} рядов")
    print(f"future_exog колонок: {len(FUTURE_EXOG_COLS)}: {FUTURE_EXOG_COLS}")
    return df, cat_to_id, ch_to_id, last_value, global_last


def step_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dates = sorted(df["_date"].unique())
    test_dates = set(dates[-TEST_SIZE:])
    train_dates = set(dates[:-TEST_SIZE])
    n_val = max(1, int(len(train_dates) * TRAIN_CONFIG["val_fraction"]))
    train_dates_sorted = sorted(train_dates)
    val_dates = set(train_dates_sorted[-n_val:])
    pure_train = set(train_dates_sorted[:-n_val])
    df_train = df[df["_date"].isin(pure_train)].copy()
    df_val = df[df["_date"].isin(val_dates)].copy()
    df_test = df[df["_date"].isin(test_dates)].copy()
    return df_train, df_val, df_test



def step_scale(
    df_train: pd.DataFrame,
    df_val:   pd.DataFrame,
    df_test:  pd.DataFrame,
    feature_names: list[str]) -> tuple:

    series_log_stats: dict = {}
    for (cat, ch), grp in df_train.groupby(["_category", "_channel"]):
        y_s = grp[TARGET_COL].astype(float).values
        mu = float(np.mean(y_s))
        sig = float(np.std(y_s)) if np.std(y_s) > 1e-8 else 1.0
        series_log_stats[(cat, ch)] = {"mean": mu, "std": sig}
    all_log = df_train[TARGET_COL].astype(float).values  # уже log1p
    global_mu = float(np.mean(all_log))
    global_sig = float(np.std(all_log)) if np.std(all_log) > 1e-8 else 1.0

    def _normalize(df: pd.DataFrame) -> np.ndarray:
        y_vals = df[TARGET_COL].astype(float).values  # уже log1p
        cats = df["_category"].values
        chs = df["_channel"].values
        mus = np.array([series_log_stats.get((c, h), {"mean": global_mu})["mean"] for c, h in zip(cats, chs)], dtype=np.float64)
        sigs = np.array([series_log_stats.get((c, h), {"std": global_sig})["std"] for c, h in zip(cats, chs)], dtype=np.float64)
        return ((y_vals-mus)/sigs).astype(np.float32)

    y_train_sc = _normalize(df_train)
    y_val_sc = _normalize(df_val)
    y_test_sc = _normalize(df_test)
    scaler_x = SequenceScaler(feature_range=(-1, 1))
    avail = [f for f in feature_names if f in df_train.columns]
    X_train = df_train[avail].fillna(0).values
    X_val = df_val[avail].fillna(0).values
    X_test = df_test[avail].fillna(0).values
    X_train_sc = scaler_x.fit_transform(X_train)
    X_val_sc = scaler_x.transform(X_val)
    X_test_sc = scaler_x.transform(X_test)

    cat_train = df_train["_cat_id"].values.astype(np.int64)
    ch_train  = df_train["_ch_id"].values.astype(np.int64)
    cat_val = df_val["_cat_id"].values.astype(np.int64)
    ch_val = df_val["_ch_id"].values.astype(np.int64)
    cat_test = df_test["_cat_id"].values.astype(np.int64)
    ch_test = df_test["_ch_id"].values.astype(np.int64)


    return (
        X_train_sc, y_train_sc, cat_train, ch_train,
        X_val_sc,   y_val_sc,   cat_val,   ch_val,
        X_test_sc,  y_test_sc,  cat_test,  ch_test,
        series_log_stats, scaler_x, avail,
        global_mu, global_sig)


def step_make_loaders(
    X_train, y_train, cat_train, ch_train,
    X_val,   y_val,   cat_val,   ch_val,) -> tuple[DataLoader, DataLoader, int]:
    window = TRAIN_CONFIG["window_size"]
    horizon = HORIZON
    bs = TRAIN_CONFIG["batch_size"]

    ds_train = TimeSeriesDataset(X_train, y_train, cat_train, ch_train,window=window, horizon=horizon)
    ds_val = TimeSeriesDataset(X_val,y_val, cat_val, ch_val,window=window, horizon=horizon)

    if len(ds_train) == 0:
        raise ValueError(
            "Нет обучающих последовательностей")

    if len(ds_val) == 0:
        print(f"Val содержит 0 последовательностей (нужно >= {window+horizon} "
              f"точек на ряд). Early stopping будет по train loss.")

    loader_train = DataLoader(ds_train, batch_size=bs, shuffle=True,drop_last=True, num_workers=0)
    loader_val   = DataLoader(ds_val,   batch_size=bs, shuffle=False,num_workers=0)
    return loader_train, loader_val, len(ds_train)


def step_train_model(
    model:nn.Module,
    loader_train: DataLoader,
    loader_val:DataLoader,
    variant_name: str) -> tuple[nn.Module, list[float], list[float]]:

    model = model.to(DEVICE)
    criterion = nn.HuberLoss(delta=1.0)
    optimizer = torch.optim.AdamW(model.parameters(),lr=TRAIN_CONFIG["lr"],weight_decay=TRAIN_CONFIG["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=10, factor=0.5, min_lr=1e-5)
    es = EarlyStopping(patience=TRAIN_CONFIG["patience"], min_delta=1e-5)
    train_losses, val_losses = [], []
    best_epoch = 0
    for epoch in range(1, TRAIN_CONFIG["max_epochs"] + 1):
        model.train()
        train_loss = 0.0
        for X_b, cat_b, ch_b, y_b in loader_train:
            X_b = X_b.to(DEVICE)
            cat_b = cat_b.to(DEVICE)
            ch_b = ch_b.to(DEVICE)
            y_b = y_b.to(DEVICE)
            optimizer.zero_grad()
            pred = model(X_b, cat_b, ch_b)
            loss = criterion(pred, y_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), TRAIN_CONFIG["grad_clip"])
            optimizer.step()
            train_loss += loss.item()
        train_loss /= max(len(loader_train), 1)
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_b, cat_b, ch_b, y_b in loader_val:
                X_b = X_b.to(DEVICE)
                cat_b = cat_b.to(DEVICE)
                ch_b = ch_b.to(DEVICE)
                y_b = y_b.to(DEVICE)
                pred = model(X_b, cat_b, ch_b)
                val_loss += criterion(pred, y_b).item()
        val_loss /= max(len(loader_val), 1)
        _no_val = len(loader_val) == 0
        es_loss = train_loss if _no_val else val_loss
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        scheduler.step(es_loss)
        if epoch % 20 == 0 or epoch == 1:
            lr_cur = optimizer.param_groups[0]["lr"]
            label  = "train(ES)" if _no_val else "val"
            print(f"  Epoch {epoch:>3}: train={train_loss:.4f}  "
                  f"{label}={es_loss:.4f}  lr={lr_cur:.2e}")

        if es.step(es_loss, model):
            best_epoch = epoch - TRAIN_CONFIG["patience"]
            print(f"Early stopping на эпохе {epoch}. "
                  f"Лучшая: {best_epoch}  {'train' if _no_val else 'val'}_loss={es.best_loss:.4f}")
            break

    es.restore_best(model)
    model.eval()

    actual_best = val_losses.index(min(val_losses)) + 1
    print(f"  Обучение завершено. Лучшая эпоха: {actual_best}  "
          f"val_loss={min(val_losses):.4f}")
    return model, train_losses, val_losses


def step_evaluate(
    model: nn.Module,
    df_test:pd.DataFrame,
    df_pretrain: pd.DataFrame,
    feature_names: list[str],
    series_log_stats: dict,
    global_mu:float,
    global_sig: float,
    scaler_x: SequenceScaler,
    cat_to_id: dict,
    ch_to_id: dict,
    last_value: dict,
    global_last: dict,
    has_future_exog: bool = False) -> dict:

    model.eval()
    window = TRAIN_CONFIG["window_size"]
    n_feats = len(feature_names)
    results = {}
    future_indices = {
        f: feature_names.index(f)
        for f in FUTURE_EXOG_COLS
        if f in feature_names}

    for (cat, ch), grp_te in df_test.groupby(["_category", "_channel"]):
        tr_s = df_pretrain[(df_pretrain["_category"] == cat) & (df_pretrain["_channel"] == ch)].copy()
        if len(tr_s) < window:
            continue
        te_dates_sorted = sorted(grp_te["_date"].unique())
        if len(te_dates_sorted) < 1:
            continue

        y_true_raw = []
        for fd in te_dates_sorted:
            fact = grp_te[grp_te["_date"] == fd]
            if not fact.empty:
                y_true_raw.append(float(fact[TARGET_COL].values[0]))
        if not y_true_raw:
            continue
        y_true = np.maximum(np.expm1(np.array(y_true_raw)), 0)
        avail_feats = [f for f in feature_names if f in tr_s.columns]
        x_window_raw = tr_s[avail_feats].fillna(0).values[-window:]
        if x_window_raw.shape[1] < n_feats:
            pad = np.zeros((window, n_feats - x_window_raw.shape[1]), dtype=np.float32)
            x_window_raw = np.hstack([x_window_raw, pad])

        x_window_sc = scaler_x.transform(x_window_raw)
        x_tensor = torch.FloatTensor(x_window_sc).unsqueeze(0).to(DEVICE)
        cat_t = torch.LongTensor([cat_to_id.get(cat, 0)]).to(DEVICE)
        ch_t  = torch.LongTensor([ch_to_id.get(ch, 0)]).to(DEVICE)
        if has_future_exog and future_indices:
            x_no = x_tensor.clone()
            for col in VIF_EXOG_VARS:
                fc  = safe_col(col) + "_future"
                idx = future_indices.get(fc)
                if idx is None:
                    continue
                lv  = last_value.get((cat, ch), {}).get(col, global_last.get(col, 0.0))
                if hasattr(scaler_x.scaler, 'scale_') and idx < len(scaler_x.scaler.scale_):
                    lv_sc = (lv - scaler_x.scaler.data_min_[idx]) * scaler_x.scaler.scale_[idx] + (-1)
                    lv_sc = float(np.clip(lv_sc, -1, 1))
                else:
                    lv_sc = 0.0
                x_no[:, -1, idx] = lv_sc
        else:
            x_no = x_tensor

        with torch.no_grad():
            y_pred_sc_no = model(x_no, cat_t, ch_t).cpu().numpy().ravel()
        stat = series_log_stats.get((cat, ch), {"mean": global_mu, "std": global_sig})
        mu, sig = stat["mean"], stat["std"]
        y_log_no = y_pred_sc_no * sig + mu
        y_pred_no = np.maximum(np.expm1(y_log_no), 0)
        n = min(len(y_true), len(y_pred_no))
        m_no = compute_metrics(y_true[:n], y_pred_no[:n])
        results_row = {"no_exog": m_no}
        if has_future_exog and future_indices:
            x_full = x_tensor.clone()
            for col in VIF_EXOG_VARS:
                fc  = safe_col(col) + "_future"
                idx = future_indices.get(fc)
                if idx is None:
                    continue
                first_test_val = last_value.get((cat, ch), {}).get(col, 0.0)
                if te_dates_sorted:
                    fact_first = grp_te[grp_te["_date"] == te_dates_sorted[0]]
                    if not fact_first.empty and col in fact_first.columns:
                        v = fact_first[col].values[0]
                        if not np.isnan(v):
                            first_test_val = float(v)
                if idx < n_feats and hasattr(scaler_x.scaler, 'scale_') and idx < len(scaler_x.scaler.scale_):
                    v_sc = (first_test_val - scaler_x.scaler.data_min_[idx]) * scaler_x.scaler.scale_[idx] + (-1)
                    v_sc = float(np.clip(v_sc, -1, 1))
                else:
                    v_sc = 0.0
                x_full[:, -1, idx] = v_sc

            with torch.no_grad():
                y_pred_sc_full = model(x_full, cat_t, ch_t).cpu().numpy().ravel()
            y_log_full  = y_pred_sc_full * sig + mu
            y_pred_full = np.maximum(np.expm1(y_log_full), 0)
            m_full = compute_metrics(y_true[:n], y_pred_full[:n])
            results_row["full_exog"] = m_full

        results[(cat, ch)] = results_row

    no_exog_mapes = [r["no_exog"]["mape"]   for r in results.values() if "no_exog"   in r]
    full_exog_mapes = [r["full_exog"]["mape"]  for r in results.values() if "full_exog" in r]

    summary = {
        "mape_no_exog_median":   round(float(np.median(no_exog_mapes)), 2) if no_exog_mapes else None,
        "mape_no_exog_mean":     round(float(np.mean(no_exog_mapes)),   2) if no_exog_mapes else None}
    if full_exog_mapes:
        summary["mape_full_exog_median"] = round(float(np.median(full_exog_mapes)), 2)
        summary["mape_full_exog_mean"]   = round(float(np.mean(full_exog_mapes)),   2)

    return {"per_series": results, "summary": summary}


def step_ablation(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    feature_names: list[str],
    cat_to_id:dict,
    ch_to_id: dict,
    n_categories: int,
    n_channels: int,
    last_value: dict,
    global_last: dict,) -> pd.DataFrame:
    groups = {"Все признаки (baseline)": list(feature_names)}
    for col in VIF_EXOG_VARS:
        lag_cols  = [f"{safe_col(col)}_lag1", f"{safe_col(col)}_lag2"]
        remaining = [f for f in feature_names if f not in lag_cols]
        groups[f"Без лагов {col}"] = remaining
    groups["Без всех exog_lags"] = [
        f for f in feature_names
        if not any(f.endswith(f"_{s}") for s in ["lag1", "lag2"])
        or f.startswith("y_lag")]

    df_pretrain = pd.concat([df_train, df_val], ignore_index=True).sort_values(["_category", "_channel", "_date"]).reset_index(drop=True)
    rows = []
    baseline_mape = None
    for name, feats in groups.items():
        avail = [f for f in feats if f in df_train.columns]
        sls_ab: dict = {}
        for (_c, _h), _grp in df_train.groupby(["_category", "_channel"]):
            _ys = np.log1p(np.maximum(_grp[TARGET_COL].values, 0))
            sls_ab[(_c, _h)] = {"mean": float(_ys.mean()),"std":  float(_ys.std()) if _ys.std() > 1e-8 else 1.0}
        _all = np.log1p(np.maximum(df_train[TARGET_COL].values, 0))
        gmu_ab = float(_all.mean()); gsg_ab = float(_all.std() or 1.0)
        def _nm(df_: pd.DataFrame) -> np.ndarray:
            out_ = np.zeros(len(df_), dtype=np.float32)
            for i_, (_, r_) in enumerate(df_.iterrows()):
                s_ = sls_ab.get((r_["_category"], r_["_channel"]),{"mean": gmu_ab, "std": gsg_ab})
                out_[i_] = (np.log1p(max(float(r_[TARGET_COL]),0))-s_["mean"])/s_["std"]
            return out_
        scaler_x_ab = SequenceScaler(feature_range=(-1, 1))
        y_tr_sc = _nm(df_train)
        y_vl_sc = _nm(df_val)
        X_tr = df_train[avail].fillna(0).values
        X_vl = df_val[avail].fillna(0).values
        X_tr_sc = scaler_x_ab.fit_transform(X_tr)
        X_vl_sc = scaler_x_ab.transform(X_vl)

        cat_tr = df_train["_cat_id"].values.astype(np.int64)
        ch_tr  = df_train["_ch_id"].values.astype(np.int64)
        cat_vl = df_val["_cat_id"].values.astype(np.int64)
        ch_vl  = df_val["_ch_id"].values.astype(np.int64)

        loader_tr, loader_vl, _ = step_make_loaders(
            X_tr_sc, y_tr_sc, cat_tr, ch_tr,
            X_vl_sc, y_vl_sc, cat_vl, ch_vl)

        model_ab = LSTMForecaster(
            input_size=len(avail),
            n_categories=n_categories,
            n_channels=n_channels,
            **{k: v for k, v in LSTM_CONFIG.items() if k != "horizon"},
            horizon=HORIZON)
        model_ab, _, _ = step_train_model(model_ab, loader_tr, loader_vl, f"ablation: {name[:30]}")

        eval_res = step_evaluate(
            model_ab, df_test, df_pretrain,
            avail, sls_ab, gmu_ab, gsg_ab, scaler_x_ab,
            cat_to_id, ch_to_id,
            last_value, global_last,
            has_future_exog=False)
        mape = eval_res["summary"].get("mape_no_exog_median", np.nan)
        if name == "Все признаки (baseline)":
            baseline_mape = mape
        delta = round(mape - baseline_mape, 2) if baseline_mape is not None else 0.0
        print(f"  Test MAPE: {mape:.2f}%  Δ={delta:+.2f}%")
        rows.append({
            "Вариант": name,
            "N признаков": len(avail),
            "MAPE,%": round(mape, 2),
            "Δ от baseline": delta})

    df_abl = pd.DataFrame(rows)
    print(df_abl.to_string(index=False))
    out = RESULTS3 / "lstm_ablation.csv"
    df_abl.to_csv(out, index=False)
    return df_abl


def step_save(
    model: nn.Module,
    series_log_stats: dict,
    global_mu:float,
    global_sig:float,
    scaler_x: SequenceScaler,
    feature_names: list[str],
    cat_to_id:dict,
    ch_to_id: dict,
    last_value:dict,
    global_last:dict,
    train_losses: list[float],
    val_losses:list[float],
    eval_summary:dict,
    variant_name:str,
    has_future_exog: bool) -> Path:
    ckpt_path = DL_DIR / f"lstm_{variant_name}.pt"
    lv_json = {f"{k[0]}|{k[1]}": v for k, v in last_value.items()}
    torch.save({
        "model_state_dict": model.state_dict(),
        "model_config": model.get_config(),
        "series_log_stats": {f"{k[0]}|{k[1]}": v for k, v in series_log_stats.items()},
        "global_mu": global_mu,
        "global_sig": global_sig,
        "scaler_x": scaler_x,
        "feature_names":feature_names,
        "cat_encoder": cat_to_id,
        "ch_encoder":ch_to_id,
        "last_value_per_series": lv_json,
        "global_last":global_last,
        "has_future_exog":has_future_exog,
        "window_size":TRAIN_CONFIG["window_size"],
        "horizon":HORIZON,
        "vif_exog_vars":VIF_EXOG_VARS,
        "future_exog_cols": FUTURE_EXOG_COLS,
        "train_losses":train_losses,
        "val_losses":val_losses,
        "eval_summary":eval_summary,
        "variant_name":variant_name}, ckpt_path)
    return ckpt_path

def step_log_mlflow(
    variant_name: str,
    eval_summary: dict,
    model_config: dict,
    val_losses:list[float],
    ckpt_path:Path,
    has_future_exog: bool,
    eval_result: dict = None):
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT_LSTM)
    except Exception as e:
        print(f"MLflow недоступен ({MLFLOW_TRACKING_URI}): {e}")
        return
    try:
        with mlflow.start_run(run_name=f"lstm_{variant_name}") as run:
            mlflow.set_tag("variant", variant_name)
            mlflow.set_tag("has_future_exog", str(has_future_exog))
            mlflow.set_tag("model_class", model_config.get("model_class", ""))
            mlflow.set_tag("device", str(DEVICE))
            mlflow.set_tag("part","3_lstm")
            mlflow.log_params({
                "hidden_size": model_config.get("hidden_size"),
                "n_layers": model_config.get("n_layers"),
                "embed_dim":model_config.get("embed_dim"),
                "dropout":model_config.get("dropout"),
                "window_size":TRAIN_CONFIG["window_size"],
                "batch_size": TRAIN_CONFIG["batch_size"],
                "lr":TRAIN_CONFIG["lr"],
                "weight_decay": TRAIN_CONFIG["weight_decay"],
                "patience":TRAIN_CONFIG["patience"],
                "n_features":model_config.get("input_size")})
            global_metrics = {}
            if eval_summary.get("mape_no_exog_median") is not None:
                global_metrics["test_mape_no_exog_median"] = eval_summary["mape_no_exog_median"]
                global_metrics["test_mape_no_exog_mean"] = eval_summary["mape_no_exog_mean"]
            if eval_summary.get("mape_full_exog_median") is not None:
                global_metrics["test_mape_full_exog_median"] = eval_summary["mape_full_exog_median"]
            if val_losses:
                global_metrics["best_val_loss"] = min(val_losses)
                global_metrics["n_epochs"] = len(val_losses)
            if global_metrics:
                mlflow.log_metrics(global_metrics)
            per_series = (eval_result or {}).get("per_series", {})
            if per_series:
                series_metrics = {}
                for (cat, ch), res in per_series.items():
                    key = (f"{cat}__{ch}".replace(" ", "_").replace("/", "-").replace("|", "__"))
                    if "no_exog" in res:
                        m = res["no_exog"].get("mape")
                        if m is not None and not np.isnan(float(m)):
                            series_metrics[f"{key}__mape"] = round(float(m), 3)
                    if "full_exog" in res:
                        m = res["full_exog"].get("mape")
                        if m is not None and not np.isnan(float(m)):
                            series_metrics[f"{key}__mape_full"] = round(float(m), 3)

                items = list(series_metrics.items())
                for i in range(0, len(items), 90):
                    mlflow.log_metrics(dict(items[i:i + 90]))

            if ckpt_path.exists():
                mlflow.log_artifact(str(ckpt_path))

    except Exception as e:
        print(f"MLflow logging failed для {variant_name}: {e}")



def step_save_per_series(
    variant_name:str,
    eval_result:dict,   # результат step_evaluate
    has_future_exog: bool) -> Path:
    per_series = eval_result.get("per_series", {})
    rows = []
    for (cat, ch), res in per_series.items():
        row = {
            "category":cat,
            "channel": ch,
            "variant":variant_name,
            "mape_no_exog":   None,
            "mape_full_exog": None}
        if "no_exog" in res:
            row["mape_no_exog"] = round(float(res["no_exog"].get("mape", np.nan)), 2)
        if "full_exog" in res:
            row["mape_full_exog"] = round(float(res["full_exog"].get("mape", np.nan)), 2)
        rows.append(row)
    df = pd.DataFrame(rows)
    out = RESULTS3 / f"lstm_{variant_name}_per_series.csv"
    df.to_csv(out, index=False)
    return out

MLFLOW_EXPERIMENT_LSTM_PER_SERIES = "mars_lstm_per_series"

def step_log_per_series_mlflow(
    variant_name:    str,
    eval_result:     dict,
    has_future_exog: bool) -> None:

    per_series = eval_result.get("per_series", {})
    if not per_series:
        print(f"step_log_per_series_mlflow: per_series пустой для {variant_name}")
        return
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT_LSTM_PER_SERIES)
    except Exception as e:
        print(f"MLflow недоступен: {e}")
        return

    logged = 0
    no_ex_mapes: list[float] = []
    try:
        for (cat, ch), res in per_series.items():
            run_name = f"{cat}|{ch}"
            with mlflow.start_run(run_name=run_name):
                mlflow.set_tag("category", cat)
                mlflow.set_tag("channel",ch)
                mlflow.set_tag("variant", variant_name)
                mlflow.set_tag("has_future_exog", str(has_future_exog))
                mlflow.set_tag("model_class","lstm")
                metrics: dict[str, float] = {}
                if "no_exog" in res:
                    mape = res["no_exog"].get("mape")
                    if mape is not None and not np.isnan(float(mape)):
                        m = round(float(mape), 3)
                        metrics["lstm_mape_no_exog"] = m
                        no_ex_mapes.append(m)
                    mae = res["no_exog"].get("mae")
                    if mae is not None and not np.isnan(float(mae)):
                        metrics["lstm_mae_no_exog"] = round(float(mae), 1)
                if "full_exog" in res:
                    mape = res["full_exog"].get("mape")
                    if mape is not None and not np.isnan(float(mape)):
                        metrics["lstm_mape_full_exog"] = round(float(mape), 3)
                if metrics:
                    mlflow.log_metrics(metrics)
                mlflow.log_params({
                    "variant": variant_name,
                    "evaluation":"h1_12_from_T",
                    "has_future_exog": str(has_future_exog)})

            logged+=1

        with mlflow.start_run(run_name=f"_summary_{variant_name}"):
            mlflow.set_tag("run_type", "summary")
            mlflow.set_tag("variant",  variant_name)
            summary_m: dict[str, float] = {"total_series": float(len(per_series))}
            if no_ex_mapes:
                summary_m["median_mape_no_exog"] = round(float(np.median(no_ex_mapes)),3)
                summary_m["mean_mape_no_exog"] = round(float(np.mean(no_ex_mapes)),3)
            mlflow.log_metrics(summary_m)

        if no_ex_mapes:
            print(f"Медиана MAPE (no_exog): {np.median(no_ex_mapes):.2f}%")

    except Exception as e:
        print(f"MLflow per-series logging failed: {e}")

def step_compare(all_results: dict, base_mape_sarima: float = None) -> pd.DataFrame:
    rows = []
    for variant, res in all_results.items():
        s = res.get("summary", {})
        row = {"Вариант": variant}
        if s.get("mape_no_exog_median") is not None:
            row["no_exog MAPE,%"]   = s["mape_no_exog_median"]
        if s.get("mape_full_exog_median") is not None:
            row["full_exog MAPE,%"] = s["mape_full_exog_median"]
        rows.append(row)
    df_cmp = pd.DataFrame(rows)

    if base_mape_sarima:
        df_cmp = pd.concat([df_cmp,pd.DataFrame([{"Вариант": "SARIMA/SARIMAX","no_exog MAPE,%": base_mape_sarima}])], ignore_index=True)
    print(df_cmp.to_string(index=False))

    if "no_exog MAPE,%" in df_cmp.columns:
        best_idx = df_cmp["no_exog MAPE,%"].dropna().idxmin()
        best_row = df_cmp.loc[best_idx]

    out = RESULTS3 / "lstm_comparison.csv"
    df_cmp.to_csv(out, index=False)
    return df_cmp


def step_select_prod_variant(all_results: dict) -> str:
    best_variant = None
    best_mape = float("inf")
    for variant, res in all_results.items():
        mape = res.get("summary", {}).get("mape_no_exog_median")
        if mape is not None and mape < best_mape:
            best_mape = mape
            best_variant = variant
    if best_variant is None:
        print("Не удалось определить лучший вариант")
        return "C_lstm_attn_exog"
    variant_clean = best_variant.split("_", 1)[-1]
    ckpt_name = f"{variant_clean}"

    meta = {
        "prod_variant": ckpt_name,
        "prod_variant_full": best_variant,
        "mape_no_exog": best_mape,
        "all_variants": {v: res.get("summary", {}) for v, res in all_results.items()}}
    meta_path = RESULTS3 / "lstm_prod_metadata.json"
    import json as _json
    with open(meta_path, "w") as f:
        _json.dump(meta, f, ensure_ascii=False, indent=2)
    return ckpt_name


def step_cross_model_comparison(all_results: dict) -> pd.DataFrame:
    results_dir = RESULTS3.parent
    sarima_path = results_dir / "part1" / "sarima_all_models_comparison.csv"
    df_sarima   = pd.DataFrame()
    if sarima_path.exists():
        df_sarima = pd.read_csv(sarima_path)
        col_map = {}
        for c in df_sarima.columns:
            if "Категория" in c or c.lower() == "category":
                col_map[c] = "category"
            elif "Канал" in c or c.lower() == "channel":
                col_map[c] = "channel"
            elif "Лучшая" in c:
                col_map[c] = "best_econometric"
            elif "SARIMAX" in c and "MAPE" in c:
                col_map[c] = "sarimax_mape"
            elif "SARIMA" in c and "MAPE" in c and "SARIMAX" not in c:
                col_map[c] = "sarima_mape"
            elif "Prophet" in c and "MAPE" in c:
                col_map[c] = "prophet_mape"
            elif "ARIMA" in c and "MAPE" in c and "SARIMA" not in c:
                col_map[c] = "arima_mape"
        df_sarima = df_sarima.rename(columns=col_map)
        eco_cols = [c for c in ["sarimax_mape", "sarima_mape", "prophet_mape", "arima_mape"] if c in df_sarima.columns]
        if eco_cols:
            df_sarima["best_eco_mape"] = df_sarima[eco_cols].min(axis=1)
        print(f"SARIMA: {len(df_sarima)} рядов из {sarima_path.name}")

    ml_path = results_dir / "part2" / "test_metrics_h12.csv"
    df_ml = pd.DataFrame()
    if ml_path.exists():
        df_ml = pd.read_csv(ml_path)
        best_ml_rows = (df_ml.groupby(["category", "channel"])["mape"].min().reset_index().rename(columns={"mape": "best_ml_mape"}))
        best_ml_model = (
            df_ml.loc[df_ml.groupby(["category", "channel"])["mape"].idxmin()]
            [["category", "channel", "model"]]
            .rename(columns={"model": "best_ml_model"}))
        df_ml = best_ml_rows.merge(best_ml_model, on=["category", "channel"], how="left")
        print(f"ML: {len(df_ml)} рядов из {ml_path.name}")
    else:
        print(f"{ml_path} не найден, ML не включена в сравнение")

    best_lstm_variant = None
    best_mape_lstm = float("inf")
    for v, res in all_results.items():
        m = res.get("summary", {}).get("mape_no_exog_median")
        if m is not None and m < best_mape_lstm:
            best_mape_lstm = m
            best_lstm_variant = v

    _KEY_TO_VARIANT = {
        "A_lstm_base": "base",
        "B_lstm_attn_base": "attn_base",
        "C_lstm_attn_exog": "attn_exog",
        "C_lstm_attn_no_exog":"attn_exog",
        "D_lstm_attn_full_exog":  "attn_exog",
        "D_lstm_exog":"exog",
        "E_lstm_exog_no_exog": "exog",
        "F_lstm_exog_full_exog": "exog"}

    df_lstm = pd.DataFrame()
    if best_lstm_variant:
        variant_name = _KEY_TO_VARIANT.get(best_lstm_variant)
        if variant_name is None:
            raw = best_lstm_variant.split("_", 1)[-1]
            variant_name = raw[len("lstm_"):] if raw.startswith("lstm_") else raw

        lstm_ps_path = RESULTS3 / f"lstm_{variant_name}_per_series.csv"
        if lstm_ps_path.exists():
            df_lstm = pd.read_csv(lstm_ps_path)[["category", "channel", "mape_no_exog"]]
            df_lstm = df_lstm.rename(columns={"mape_no_exog": "lstm_mape"})
            df_lstm["best_lstm_variant"] = variant_name
            print(f"  LSTM: {len(df_lstm)} рядов из {lstm_ps_path.name}")
        else:
            print(f"{lstm_ps_path} не найден, берём из all_results (per_series)")
            per_series = all_results[best_lstm_variant].get("per_series", {})
            if not per_series:
                for full_key in ["C_lstm_attn_exog", "D_lstm_exog", "A_lstm_base", "B_lstm_attn_base"]:
                    per_series = all_results.get(full_key, {}).get("per_series", {})
                    if per_series:
                        variant_name = _KEY_TO_VARIANT.get(full_key, full_key)
                        break
            rows_lstm = []
            for (cat, ch), res in per_series.items():
                mape = res.get("no_exog", {}).get("mape")
                rows_lstm.append({"category": cat, "channel": ch,
                    "lstm_mape": round(float(mape), 2) if mape is not None else None,
                    "best_lstm_variant": variant_name})
            df_lstm = pd.DataFrame(rows_lstm)
            if df_lstm.empty:
                print("per_series пустой — LSTM не попадёт в сравнение")

    df_all = pd.DataFrame()
    for df_part, on_cols in [(df_sarima, ["category", "channel"]),
        (df_ml, ["category", "channel"]),
        (df_lstm, ["category", "channel"])]:
        if df_part.empty:
            continue
        if df_all.empty:
            df_all = df_part
        else:
            keep  = on_cols + [c for c in df_part.columns if c not in df_all.columns]
            df_all = df_all.merge(df_part[keep], on=on_cols, how="outer")

    if df_all.empty:
        print("Нет данных для cross-model сравнения")
        return pd.DataFrame()

    mape_cols = [c for c in ["best_eco_mape", "best_ml_mape", "lstm_mape"] if c in df_all.columns]
    label_map = {"best_eco_mape": "Эконометрика", "best_ml_mape": "ML", "lstm_mape": "LSTM"}

    def best_model(row):
        vals = {label_map[c]: row[c] for c in mape_cols if pd.notna(row.get(c))}
        return min(vals, key=vals.get) if vals else "—"
    df_all["Победитель"] = df_all.apply(best_model, axis=1)
    out = RESULTS3 / "cross_model_comparison_per_series.csv"
    df_all.to_csv(out, index=False)
    if mape_cols:
        for c in mape_cols:
            if c in df_all.columns:
                med = df_all[c].dropna().median()
                print(f"  {label_map[c]:<15}: медиана MAPE = {med:.2f}%")

    if "Победитель" in df_all.columns:
        print("Победитель по рядам:")
        for lbl, cnt in df_all["Победитель"].value_counts().items():
            print(f" {lbl}: {cnt} рядов")
    return df_all

def main():
    (df_raw, series_dict, global_df, feature_cols,
     cat_enc, ch_enc, lag_ranges, categories, channels) = step_load_data()
    n_categories = len(categories)
    n_channels = len(channels)

    df_full, cat_to_id, ch_to_id, last_value, global_last = step_build_features(
        global_df, series_dict, categories, channels)

    df_train, df_val, df_test = step_split(df_full)
    df_pretrain = pd.concat([df_train, df_val], ignore_index=True).sort_values(["_category", "_channel", "_date"]).reset_index(drop=True)
    all_results = {}
    base_feats = get_base_feature_names()
    (X_tr, y_tr, cat_tr, ch_tr,
     X_vl, y_vl, cat_vl, ch_vl,
     X_te, y_te, cat_te, ch_te,
     sls_a, sc_x_a, avail_a,
     gmu_a, gsg_a) = step_scale(df_train, df_val, df_test, base_feats)
    loader_tr_a, loader_vl_a, _ = step_make_loaders(
        X_tr, y_tr, cat_tr, ch_tr, X_vl, y_vl, cat_vl, ch_vl)

    model_base = LSTMForecaster(input_size=len(avail_a), n_categories=n_categories, n_channels=n_channels,
        **{k: v for k, v in LSTM_CONFIG.items() if k != "horizon"}, horizon=HORIZON)
    model_base, tl_a, vl_a = step_train_model(model_base, loader_tr_a, loader_vl_a, "lstm_base")
    eval_a = step_evaluate(
        model_base, df_test, df_pretrain, avail_a,
        sls_a, gmu_a, gsg_a, sc_x_a, cat_to_id, ch_to_id, last_value, global_last,
        has_future_exog=False)
    print(f"\n  A (LSTM base) no_exog MAPE: {eval_a['summary'].get('mape_no_exog_median', '—')}%")
    all_results["A_lstm_base"] = eval_a
    ckpt_a = step_save(
        model_base, sls_a, gmu_a, gsg_a, sc_x_a, avail_a,
        cat_to_id, ch_to_id, last_value, global_last,
        tl_a, vl_a, eval_a["summary"], "base", has_future_exog=False)
    step_log_mlflow("base", eval_a["summary"], model_base.get_config(),vl_a, ckpt_a, has_future_exog=False, eval_result=eval_a)
    step_save_per_series("base", eval_a, has_future_exog=False)
    step_log_per_series_mlflow("base", eval_a, has_future_exog=False)

    model_attn_base = LSTMAttentionForecaster(
        input_size=len(avail_a), n_categories=n_categories, n_channels=n_channels,
        **{k: v for k, v in LSTM_CONFIG.items() if k != "horizon"}, horizon=HORIZON)
    model_attn_base, tl_b, vl_b = step_train_model(
        model_attn_base, loader_tr_a, loader_vl_a, "lstm_attn_base")
    eval_b = step_evaluate(
        model_attn_base, df_test, df_pretrain, avail_a,
        sls_a, gmu_a, gsg_a, sc_x_a, cat_to_id, ch_to_id, last_value, global_last,
        has_future_exog=False)
    print(f"\n  B (LSTM Attn base) no_exog MAPE: {eval_b['summary'].get('mape_no_exog_median', '—')}%")
    all_results["B_lstm_attn_base"] = eval_b

    ckpt_b = step_save(
        model_attn_base, sls_a, gmu_a, gsg_a, sc_x_a, avail_a,
        cat_to_id, ch_to_id, last_value, global_last,
        tl_b, vl_b, eval_b["summary"], "attn_base", has_future_exog=False)
    step_log_mlflow("attn_base", eval_b["summary"], model_attn_base.get_config(),vl_b, ckpt_b, has_future_exog=False, eval_result=eval_b)
    step_save_per_series("attn_base", eval_b, has_future_exog=False)
    step_log_per_series_mlflow("attn_base", eval_b, has_future_exog=False)

    exog_feats = get_exog_feature_names()
    (X_tr_c, y_tr_c, cat_tr_c, ch_tr_c,
     X_vl_c, y_vl_c, cat_vl_c, ch_vl_c,
     X_te_c, y_te_c, cat_te_c, ch_te_c,
     sls_c, sc_x_c, avail_c,
     gmu_c, gsg_c) = step_scale(df_train, df_val, df_test, exog_feats)

    loader_tr_c, loader_vl_c, _ = step_make_loaders(
        X_tr_c, y_tr_c, cat_tr_c, ch_tr_c,
        X_vl_c, y_vl_c, cat_vl_c, ch_vl_c)

    model_attn_exog = LSTMAttentionForecaster(
        input_size=len(avail_c), n_categories=n_categories, n_channels=n_channels,
        **{k: v for k, v in LSTM_CONFIG.items() if k != "horizon"}, horizon=HORIZON)
    model_attn_exog, tl_c, vl_c = step_train_model(
        model_attn_exog, loader_tr_c, loader_vl_c, "lstm_attn_exog")
    eval_c = step_evaluate(
        model_attn_exog, df_test, df_pretrain, avail_c,
        sls_c, gmu_c, gsg_c, sc_x_c, cat_to_id, ch_to_id, last_value, global_last,
        has_future_exog=True)

    all_results["C_lstm_attn_exog"]  = eval_c
    all_results["C_lstm_attn_no_exog"] = {
        "summary": {"mape_no_exog_median": eval_c["summary"].get("mape_no_exog_median"),"mape_no_exog_mean":   eval_c["summary"].get("mape_no_exog_mean")},
        "per_series": eval_c.get("per_series", {})}
    all_results["D_lstm_attn_full_exog"] = {
        "summary": {"mape_no_exog_median": eval_c["summary"].get("mape_full_exog_median"),"mape_no_exog_mean":   eval_c["summary"].get("mape_full_exog_mean")},
        "per_series": eval_c.get("per_series", {})}

    ckpt_c = step_save(
        model_attn_exog, sls_c, gmu_c, gsg_c, sc_x_c, avail_c,
        cat_to_id, ch_to_id, last_value, global_last,
        tl_c, vl_c, eval_c["summary"], "attn_exog", has_future_exog=True)
    step_log_mlflow("attn_exog", eval_c["summary"], model_attn_exog.get_config(),vl_c, ckpt_c, has_future_exog=True, eval_result=eval_c)
    step_save_per_series("attn_exog", eval_c, has_future_exog=True)
    step_log_per_series_mlflow("attn_exog", eval_c, has_future_exog=True)

    model_exog = LSTMForecaster(
        input_size=len(avail_c), n_categories=n_categories, n_channels=n_channels,
        **{k: v for k, v in LSTM_CONFIG.items() if k != "horizon"}, horizon=HORIZON)
    model_exog, tl_d, vl_d = step_train_model(
        model_exog, loader_tr_c, loader_vl_c, "lstm_exog")
    eval_d = step_evaluate(
        model_exog, df_test, df_pretrain, avail_c,
        sls_c, gmu_c, gsg_c, sc_x_c, cat_to_id, ch_to_id, last_value, global_last,
        has_future_exog=True)

    all_results["D_lstm_exog"] = eval_d
    all_results["E_lstm_exog_no_exog"] = {
        "summary": {"mape_no_exog_median": eval_d["summary"].get("mape_no_exog_median"),"mape_no_exog_mean":   eval_d["summary"].get("mape_no_exog_mean")},
        "per_series": eval_d.get("per_series", {})}
    all_results["F_lstm_exog_full_exog"] = {
        "summary":    {"mape_no_exog_median": eval_d["summary"].get("mape_full_exog_median"), "mape_no_exog_mean":   eval_d["summary"].get("mape_full_exog_mean")},
        "per_series": eval_d.get("per_series", {})}

    ckpt_d = step_save(
        model_exog, sls_c, gmu_c, gsg_c, sc_x_c, avail_c,
        cat_to_id, ch_to_id, last_value, global_last,
        tl_d, vl_d, eval_d["summary"], "exog", has_future_exog=True)
    step_log_mlflow("exog", eval_d["summary"], model_exog.get_config(),vl_d, ckpt_d, has_future_exog=True, eval_result=eval_d)
    step_save_per_series("exog", eval_d, has_future_exog=True)
    step_log_per_series_mlflow("exog", eval_d, has_future_exog=True)
    try:
        step_ablation(df_train, df_val, df_test,avail_a,cat_to_id, ch_to_id,n_categories, n_channels,last_value, global_last)
    except Exception as e:
        print(f"Ablation пропущен: {e}")

    prod_variant = step_select_prod_variant(all_results)
    print(f"PROD checkpoint: models/dl/{prod_variant}.pt")
    step_cross_model_comparison(all_results)
    return {
        "model_base":model_base,
        "model_attn_base": model_attn_base,
        "model_attn_exog": model_attn_exog,
        "model_exog":model_exog,
        "results":all_results,
        "prod_variant":prod_variant}


def _get_sarima_median_mape() -> float | None:
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = mlflow.tracking.MlflowClient()
        for exp_name in ["mars-forecasting", "mars_sarima", "mars_ml_exog_v2_comparison"]:
            exp = client.get_experiment_by_name(exp_name)
            if exp is None:
                continue
            runs = client.search_runs(experiment_ids=[exp.experiment_id], max_results=2000)
            mapes = [float(r.data.metrics["best_mape"]) for r in runs if "best_mape" in r.data.metrics]
            if mapes:
                return round(float(np.median(mapes)), 2)
    except Exception:
        pass
    return None


if __name__ == "__main__":
    main()