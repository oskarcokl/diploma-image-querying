from celery import Celery

app = Celery(
    "celery_app", broker="amqp://", backend="rpc://", include=["celery_app.tasks"]
)

# Optional configuration, see the application user guide.
app.conf.update(
    result_expires=3600,
)

if __name__ == "__main__":
    app.start()
