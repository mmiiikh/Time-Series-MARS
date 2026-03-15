import os

DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": os.getenv("DB_PORT", "5432"),
    "dbname": os.getenv("DB_NAME", "mars"),
    "user": os.getenv("DB_USER", "mars_user"),
    "password": os.getenv("DB_PASSWORD", "mars_password"),
}