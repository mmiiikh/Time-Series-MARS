# Time-Series-MARS

### Требования
- [Docker](https://www.docker.com/)
- [minikube](https://minikube.sigs.k8s.io/docs/start/)
- [kubectl](https://kubernetes.io/docs/tasks/tools/)
- [DVC](https://dvc.org/) с доступом к Yandex S3

Установка на Mac:
```bash
brew install minikube kubectl
pip install dvc[s3]
```
	

### 1. Клонировать репозиторий
```bash
git clone https://github.com/mmiiikh/Time-Series-MARS.git
cd Time-Series-MARS
git checkout checkpoint_5
```

### 2. Подтянуть данные через DVC
```bash
dvc pull
```
Потребуются ключи доступа к Yandex S3 (запросите у меня)

### 3. Создать файл с секретами
Создать файл `k8s/secret.yaml` (не хранится в репозитории):
```yaml
apiVersion: v1
kind: Secret
metadata:
  name: mars-secret
type: Opaque
stringData:
  DB_PASSWORD: mars_password
  POSTGRES_PASSWORD: mars_password
  AWS_ACCESS_KEY_ID: [авторский код :) ]
  AWS_SECRET_ACCESS_KEY: [авторский код :)]
  AWS_DEFAULT_REGION: ru-central1
```

### 4. Запустить minikube
```bash
minikube start
```

### 5. Переключиться на docker внутри minikube
```bash
eval $(minikube docker-env)
```

### 6. Собрать образы
```bash
docker build -t mars-app:latest -f docker/Dockerfile .
docker build -t mars-nginx:latest -f docker/Dockerfile.nginx docker/
docker build -t mars-static:latest -f docker/Dockerfile.static .
```

### 7. Применить конфигурации
```bash
kubectl apply -f k8s/
```

### 8. Дождаться запуска всех подов
```bash
kubectl get pods -w
```
Все поды должны перейти в статус `Running`.

### 9.Получить URL сервиса(оставить терминал открытым)
```bash
minikube service nginx --url
```

### Проверка работы

В новом терминале использовать URL из предыдущего шага:
```bash
# Проверить health
curl http:///health

# Запустить обучение модели
curl -X POST http:///train

# Получить статус задачи (task_id из ответа выше)
curl http:///status/
```

Ожидаемые ответы:
```json
// GET /health
{"status": "ok"}

// POST /train
{"task_id": 1, "status": "PENDING"}

// GET /status/1
{"id": 1, "model_name": "SARIMA", "status": "SUCCESS", "mape": 2.97, "created_at": "..."}
```

### Остановка
```bash
kubectl delete -f k8s/
minikube stop
```
