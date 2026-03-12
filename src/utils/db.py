import psycopg2
from src.config.settings import DB_CONFIG
from psycopg2.extras import RealDictCursor


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
            cur.execute("""
                CREATE TABLE IF NOT EXISTS tasks (
                    id SERIAL PRIMARY KEY,
                    model_name TEXT,
                    status TEXT,
                    mape FLOAT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
        conn.commit()


def save_result(model_name: str, mape_value: float):
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO forecasts (model_name, mape) VALUES (%s, %s);",
                (model_name, mape_value),
            )

def create_task(model_name: str):
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO tasks (model_name, status) VALUES (%s, %s) RETURNING id;",
                (model_name, "PENDING"),
            )
            task_id = cur.fetchone()[0]
        conn.commit()
    return task_id


def update_task_status(task_id: int, status: str, mape: float = None):
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE tasks
                SET status = %s,
                    mape = %s
                WHERE id = %s;
                """,
                (status, mape, task_id),
            )
        conn.commit()


def get_task(task_id: int):
    with get_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "SELECT * FROM tasks WHERE id = %s;",
                (task_id,),
            )
            return cur.fetchone()