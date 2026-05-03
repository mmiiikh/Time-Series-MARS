from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset


WINDOW_SIZE = 24
HORIZON = 12


VIF_EXOG_VARS = [
    "Penetration",
    "Spend per Trip",
    "NT_CWD",
    "NT_Price per kg",
    "Volume per Trip"]

BASE_FEATURE_GROUPS = {
    "target_lags": ["y_lag1", "y_lag2", "y_lag3", "y_lag6", "y_lag12"],
    "rolling": ["rolling_mean_3", "rolling_mean_6", "rolling_std_3","rolling_min_6"],
    "ewm": ["ewm_03", "ewm_07"],
    "calendar": ["month_sin", "month_cos", "quarter_sin", "quarter_cos","is_q4", "is_summer"],
    "trend": ["t", "t_squared"],
    "covid": ["covid", "post_covid"],
    "exog_lags":     [
        "Penetration_lag1", "Penetration_lag2",
        "Spend_per_Trip_lag1", "Spend_per_Trip_lag2",
        "NT_CWD_lag1","NT_CWD_lag2",
        "NT_Price_per_kg_lag1", "NT_Price_per_kg_lag2",
        "Volume_per_Trip_lag1", "Volume_per_Trip_lag2"]}

FUTURE_EXOG_COLS = [f"{col.replace(' ', '_').replace('/', '_').replace('-', '_')}_future"
    for col in VIF_EXOG_VARS]


class TimeSeriesDataset(Dataset):

    def __init__(
        self,
        X:np.ndarray,
        y: np.ndarray,
        cat_ids: np.ndarray,
        ch_ids: np.ndarray,
        window: int = WINDOW_SIZE,
        horizon: int = HORIZON):
        assert len(X) == len(y) == len(cat_ids) == len(ch_ids), ("X, y, cat_ids, ch_ids должны иметь одинаковую длину")
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.cat_ids = torch.LongTensor(cat_ids)
        self.ch_ids = torch.LongTensor(ch_ids)
        self.window = window
        self.horizon = horizon
        self.valid_indices = self._build_valid_indices()

    def _build_valid_indices(self) -> list[int]:
        valid = []
        n = len(self.X)
        for i in range(n-self.window-self.horizon+1):
            full_cats = self.cat_ids[i: i + self.window + self.horizon]
            full_chs  = self.ch_ids[i: i + self.window + self.horizon]
            if (full_cats == full_cats[0]).all() and (full_chs  == full_chs[0]).all():
                valid.append(i)
        return valid

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int):
        i = self.valid_indices[idx]
        X_seq = self.X[i: i+self.window]
        y_seq  = self.y[i+self.window : i + self.window+self.horizon]
        cat_id = self.cat_ids[i]
        ch_id = self.ch_ids[i]
        return X_seq, cat_id, ch_id, y_seq


class SequenceScaler:

    def __init__(self, feature_range: tuple = (-1, 1)):
        from sklearn.preprocessing import MinMaxScaler
        self.scaler = MinMaxScaler(feature_range=feature_range)
        self.feature_range = feature_range
        self.is_fitted = False

    def fit(self, data: np.ndarray) -> "SequenceScaler":
        data_2d = data.reshape(-1, 1) if data.ndim == 1 else data
        self.scaler.fit(data_2d)
        self.is_fitted = True
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        assert self.is_fitted, "Scaler не обучен — вызовите fit() сначала"
        is_1d = data.ndim == 1
        data_2d = data.reshape(-1, 1) if is_1d else data
        result = self.scaler.transform(data_2d)
        return result.ravel() if is_1d else result

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        return self.fit(data).transform(data)

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        is_1d = data.ndim == 1
        data_2d = data.reshape(-1, 1) if is_1d else data
        result = self.scaler.inverse_transform(data_2d)
        return result.ravel() if is_1d else result


class EarlyStopping:

    def __init__(self, patience: int = 20, min_delta: float = 1e-5):
        self.patience = patience
        self.min_delta  = min_delta
        self.best_loss = float("inf")
        self.counter = 0
        self.best_state = None
        self.should_stop = False

    def step(self, val_loss: float, model: nn.Module) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop

    def restore_best(self, model: nn.Module) -> None:
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


class LSTMForecaster(nn.Module):

    def __init__(
        self,
        input_size: int,
        n_categories: int,
        n_channels: int,
        embed_dim: int = 4,
        hidden_size: int = 64,
        n_layers: int  = 2,
        dropout: float = 0.2,
        horizon: int  = HORIZON):
        super().__init__()
        self.input_size = input_size
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.horizon = horizon
        self.cat_embed = nn.Embedding(n_categories, embed_dim)
        self.ch_embed = nn.Embedding(n_channels,   embed_dim)
        lstm_input_size = input_size+2*embed_dim
        self.lstm = nn.LSTM(
            input_size = lstm_input_size,
            hidden_size = hidden_size,
            num_layers = n_layers,
            batch_first = True,
            dropout = dropout if n_layers > 1 else 0.0)

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, horizon)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "bias" in name:
                nn.init.zeros_(param.data)

    def forward(
        self,
        x: torch.Tensor,
        cat_id: torch.Tensor,
        ch_id:  torch.Tensor) -> torch.Tensor:

        batch, window, _ = x.shape
        cat_emb = self.cat_embed(cat_id).unsqueeze(1).expand(-1, window, -1)
        ch_emb = self.ch_embed(ch_id).unsqueeze(1).expand(-1, window, -1)
        x_aug = torch.cat([x, cat_emb, ch_emb], dim=-1)
        lstm_out, _ = self.lstm(x_aug)
        last_hidden = lstm_out[:, -1, :]
        out = self.dropout(last_hidden)
        return self.fc(out)

    def get_config(self) -> dict:
        return {
            "model_class": "LSTMForecaster",
            "input_size": self.input_size,
            "n_categories": self.cat_embed.num_embeddings,
            "n_channels":self.ch_embed.num_embeddings,
            "embed_dim":self.embed_dim,
            "hidden_size": self.hidden_size,
            "n_layers":self.n_layers,
            "dropout": self.dropout.p,
            "horizon": self.horizon}

class LSTMAttentionForecaster(nn.Module):

    def __init__(
        self,
        input_size: int,
        n_categories: int,
        n_channels: int,
        embed_dim: int = 4,
        hidden_size: int = 64,
        n_layers: int = 2,
        dropout: float = 0.2,
        horizon:int = HORIZON):
        super().__init__()
        self.input_size = input_size
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.horizon = horizon
        self.cat_embed = nn.Embedding(n_categories, embed_dim)
        self.ch_embed = nn.Embedding(n_channels,   embed_dim)
        lstm_input_size = input_size+2*embed_dim
        self.lstm = nn.LSTM(
            input_size = lstm_input_size,
            hidden_size = hidden_size,
            num_layers = n_layers,
            batch_first = True,
            dropout = dropout if n_layers > 1 else 0.0)

        self.attention = nn.Linear(hidden_size, 1, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, horizon)
        self.last_attention_weights: torch.Tensor | None = None
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)
        nn.init.xavier_uniform_(self.attention.weight)
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "bias" in name:
                nn.init.zeros_(param.data)

    def forward(
        self,
        x: torch.Tensor,
        cat_id: torch.Tensor,
        ch_id:  torch.Tensor) -> torch.Tensor:

        batch, window, _ = x.shape
        cat_emb = self.cat_embed(cat_id).unsqueeze(1).expand(-1, window, -1)
        ch_emb  = self.ch_embed(ch_id).unsqueeze(1).expand(-1, window, -1)
        x_aug = torch.cat([x, cat_emb, ch_emb], dim=-1)
        lstm_out, _ = self.lstm(x_aug)
        scores = self.attention(lstm_out)
        weights = torch.softmax(scores, dim=1)

        self.last_attention_weights = weights.detach()
        context = (weights * lstm_out).sum(dim=1)
        out = self.dropout(context)
        return self.fc(out)

    def get_attention_weights(self) -> np.ndarray | None:
        if self.last_attention_weights is None:
            return None
        return self.last_attention_weights.squeeze(-1).cpu().numpy()

    def get_config(self) -> dict:
        return {
            "model_class": "LSTMAttentionForecaster",
            "input_size": self.input_size,
            "n_categories": self.cat_embed.num_embeddings,
            "n_channels": self.ch_embed.num_embeddings,
            "embed_dim": self.embed_dim,
            "hidden_size": self.hidden_size,
            "n_layers": self.n_layers,
            "dropout":self.dropout.p,
            "horizon": self.horizon}


def build_model_from_config(config: dict) -> nn.Module:
    model_class = config.get("model_class", "LSTMForecaster")
    kwargs = {k: v for k, v in config.items() if k != "model_class"}
    if model_class == "LSTMForecaster":
        return LSTMForecaster(**kwargs)
    elif model_class == "LSTMAttentionForecaster":
        return LSTMAttentionForecaster(**kwargs)
    else:
        raise ValueError(f"Неизвестный класс модели: {model_class}")


def get_base_feature_names() -> list[str]:
    names = []
    for group in BASE_FEATURE_GROUPS.values():
        names.extend(group)
    return names


def get_exog_feature_names() -> list[str]:
    return get_base_feature_names() + FUTURE_EXOG_COLS


def safe_col(col: str) -> str:
    return col.replace(" ", "_").replace("/", "_").replace("-", "_")