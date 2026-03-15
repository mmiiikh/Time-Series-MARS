import pandas as pd
from pathlib import Path


RAW_DATA_PATH = Path("data/raw/mars_data.xlsx")


def load_raw_data() -> pd.DataFrame:
    """
    Загружает сырые данные.
    Предполагается, что файл получен через `dvc pull`.
    """
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            "Файл данных не найден. Выполните `dvc pull`.")

    df = pd.read_excel(RAW_DATA_PATH)
    return df