# Time-Series-MARS
Repository for code base of time series model to forecast sales of MARS

В рамках checkpoint 3 реализована асинхронная архитектура обучения модели SARIMA с использованием:

1) FastAPI
2) Celery(асинхронные задачи)
3) Redis(брокер сообщений)
4) PostgreSQL (хранение результатов,было в чекпоинте 2)
5) DVC + S3 (AWS)(хранение данных, было в чекпоинте 2)
6) Docker Compose (оркестрация сервисов, было в чекпоинте 2)


# Архитектура
Client-> FastAPI -> Redis -> Celery Worker -> SARIMA-> PostgreSQL

В чекпоинте3 были добавлены скрипты по celery_worker и tasks.py, также были скорректированы докер файлы и энтрипоинт для вызова соответсующи команд

# Запуск

1) Создать файл .env в папке docker/:
   DB_HOST=postgres
   
  DB_PORT=5432
  
  DB_NAME=mars
  
  DB_USER=mars_user
  
  DB_PASSWORD=mars_password
  

  AWS_ACCESS_KEY_ID=your_key
  
  AWS_SECRET_ACCESS_KEY=your_secret
  
  AWS_DEFAULT_REGION=your_region

2) Из папки docker/ выполнить:

   docker-compose down -v
   
  docker-compose up --build

  Будут запущены контейнеры: mars_postgres, mars_redis, mars_app, mars_worker

  В энтрипоинт прописан запуск dvc pull, поэтому отдельно выполнять команду не нужно.


  # Проверка работы

  Открыть Swagger: http://localhost:8000/docs

  В Swagger выполнить: POST /train

  Проверка Логов:

  docker-compose logs worker

  Ожидаемый результат:

Task received

SARIMA model trained successfully

MAPE: 2,97%

Result saved to PostgreSQL

Task succeeded

<img width="1512" height="982" alt="Снимок экрана 2026-02-26 в 21 41 21" src="https://github.com/user-attachments/assets/a5a339bb-9bfa-4937-a554-b9ff1d7faa37" />


# Проверка базы данных

docker-compose exec postgres psql -U mars_user -d mars

SELECT * FROM forecasts;

  

