import psycopg2
from src.config.settings import DB_CONFIG


def get_connection():
    return psycopg2.connect(**DB_CONFIG)


def init_tables():
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS forecasts (
                    id SERIAL PRIMARY KEY,
                    model_name TEXT,
                    mape FLOAT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)


def save_result(model_name: str, mape_value: float):
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO forecasts (model_name, mape) VALUES (%s, %s);",
                (model_name, mape_value),
            )