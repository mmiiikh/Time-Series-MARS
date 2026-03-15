# Time-Series-MARS
Repository for code base of time series model to forecast sales of MARS

В этом чекпоинте реализован сервис для обучения модели SARIMA через асинхронные задачи. Основные компоненты:

	1.	FastAPI (веб-сервис с эндпоинтами для запуска обучения и проверки статуса задачи)
	
	2.	Celery (очередь задач для асинхронного выполнения обучения модели)
	
	3.	Redis (брокер задач для Celery)
	
	4.	PostgreSQL (хранение информации о задачах и их статусе)
	
	5.	DVC (управление данными и чекпоинтами модели)
	
	6.	NGINX (прокси для FastAPI на порту 80)
	

# Как запустить

Как и в прошлых чекпоинтах нужно создать .env файл в папке docker для запуска (передать AWS и DB параметры - см. README прошлых чекпоинтов)

1. Собрать и поднять контейнеры через Docker Compose: docker-compose up --build

2. Должны быть запущены: app, worker, redis, postgres, nginx, static.

3. Отправить задачу на обучение модели SARIMA: curl -X POST http://localhost/train

	(либо можно проверить в вебе по 80 порту)

	В ответ придет JSON с task_id и статусом PENDING, например:

	{
  	"task_id": 1,
  	"status": "PENDING"
	}

4. Проверять статус задачи: curl http://localhost/status/1

   Будет один из статусов: PENDING-задача поставлена в очередь, но еще не выполнена, SUCCESS-задача выполнена успешно, FAILURE-произошла ошибка при обучении

	На вызоде должно получиться следующее:
<img width="1396" height="749" alt="Снимок экрана 2026-03-01 в 14 52 31" src="https://github.com/user-attachments/assets/65a41081-5d01-45eb-8df2-ee16c66c0e4f" />



  
