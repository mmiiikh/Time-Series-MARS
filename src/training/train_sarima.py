from src.data.load_data import load_raw_data
from src.data.preprocess import prepare_total_sales
from src.forecasting.sarima import SarimaModel
from src.utils.metrics import mape
from src.utils.db import init_tables, save_result


TEST_SIZE = 6


def main():
    df = load_raw_data()
    ts = prepare_total_sales(df)

    train = ts.iloc[:-TEST_SIZE]
    test = ts.iloc[-TEST_SIZE:]

    model = SarimaModel(
        order=(0, 1, 0),
        seasonal_order=(0, 1, 0, 12),
    )

    model.fit(train)
    forecast = model.forecast(TEST_SIZE)

    score = mape(test, forecast)

    init_tables()
    save_result("SARIMA_total_sales", score)

    print("SARIMA model trained successfully")
    print(f"MAPE: {score:.2f}%")
    print("Result saved to PostgreSQL")


if __name__ == "__main__":
    main()