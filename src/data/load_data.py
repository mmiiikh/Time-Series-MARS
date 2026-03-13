import numpy as np
import pandas as pd
from pathlib import Path
from src.config.settings import (
    TARGET_COL, DATE_COL, CATEGORY_COL, CHANNEL_COL, DATA_PATH
)


def load_raw_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"файл данных не найден: {DATA_PATH}\n"
            f"выполни: dvc pull"
        )

    df = pd.read_excel(DATA_PATH)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.dropna(subset=[TARGET_COL])
    df = df.sort_values([CATEGORY_COL, DATE_COL]).reset_index(drop=True)

    print(f"Загружено: {df.shape[0]} строк, {df.shape[1]} столбцов")
    print(f"Период: {df[DATE_COL].min().date()} — {df[DATE_COL].max().date()}")
    print(f"Категорий: {df[CATEGORY_COL].nunique()}, "
          f"каналов: {df[CHANNEL_COL].nunique()}")
    print(f"Уникальных пар: "
          f"{df.groupby([CATEGORY_COL, CHANNEL_COL]).ngroups}")
    return df


def create_series_dict(df: pd.DataFrame) -> dict:
    series_dict = {}
    for (cat, ch), grp in df.groupby([CATEGORY_COL, CHANNEL_COL]):
        grp      = grp.set_index(DATE_COL).sort_index()
        full_idx = pd.date_range(grp.index.min(), grp.index.max(), freq="MS")
        grp      = grp.reindex(full_idx)
        num_cols = grp.select_dtypes(include=np.number).columns
        grp[num_cols] = grp[num_cols].interpolate(
            method="linear", limit_direction="both"
        )
        series_dict[(cat, ch)] = grp
    return series_dict


def load_data(filepath: str = None) -> pd.DataFrame:
    return load_raw_data()