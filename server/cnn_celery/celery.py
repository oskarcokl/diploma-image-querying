from celery import Celery

app = Celery(
    "cnn_celery",
    broker="amqp://",
    backend="rpc://",
    include=["cnn_celery.cnn_tasks"],
)

# Optional configuration, see the application user guide.
app.conf.update(
    result_expires=3600,
)

if __name__ == "__main__":
    app.start()
