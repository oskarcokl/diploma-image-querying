from celery import Celery

app = Celery(
    "celery_app", broker="amqp://", backend="rpc://", include=["celery_app.tasks"]
)

app.conf.update(result_expires=3600)

if __name__ == "__main__":
    app.start()
