# Time-Series-MARS
Repository for code base of time series model to forecast sales of MARS

Проект использует SARIMA для прогнозирования тотальных продаж. Данные хранятся через **DVC** на Yandex Cloud S3, результаты сохраняются в PostgreSQL.

Требования:

-Docker и Docker Compose

-DVC (для локальной работы)

-Доступ к бакету с данными (AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_DEFAULT_REGION)

Если нужно для провери, готова передать свои секреты :) Напишите пожалуйста в тг: mmikhlv

На текущем этапе отрисована полностью структура проекта, реализована SARIMA модель для прогнозирования тотал продаж компании Марс, рассчитана метрика MAPE. В дальнейшем (уже создана структура) будут написаны скрипты под ML и DL методы.

Сейчас реализованы загрузка данных DVC, процессинг, расчет метрики MAPE, сбор SARIMA модели, настройка Docker и Docker Сompose.

# Инструкция
1. Склонировать содержимое репозитория к себе
   
2. Настроить переменные окружения:
Создать файл `.env` в папке `docker`:

docker/.env

AWS_ACCESS_KEY_ID=<ваш ключ>

AWS_SECRET_ACCESS_KEY=<ваш секрет>

AWS_DEFAULT_REGION=<регион, например ru-central1>

DB_HOST=postgres

DB_PORT=5432

DB_NAME=mars

DB_USER=mars_user

DB_PASSWORD=mars_password

4. Из папки docker поднять контейнеры и проверить что они запущены: docker-compose up --build
Дополнительно можно напсиать команду docker-compose exec app dvc pull для проверки подтягивания данных из dvc.

Как итог отработки в терминале отображается:
mars_app       | SARIMA model trained successfully
mars_app       | MAPE: 2.97%
mars_app       | Result saved to PostgreSQL

Результат сохраняется в базе PostgreSQL (table: results)

4. Чтобы остановить проект, нужно написать docker-compose down







