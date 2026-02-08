import pandas as pd

DATE_COL = "PER_SDESC"
TARGET_COL = "MT_Volume KG"


def prepare_total_sales(df: pd.DataFrame) -> pd.Series:
    """
    Агрегирует продажи по месяцам (тотал).
    """
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])

    ts = (
        df.groupby(DATE_COL)[TARGET_COL]
        .sum()
        .sort_index()
    )

    ts.name = "total_sales"
    return ts