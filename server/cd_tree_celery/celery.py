from celery import Celery

app = Celery(
    "cd_tree_celery",
    broker="amqp://",
    backend="rpc://",
    include=["cd_tree_celery.tasks", "cd_tree_celery.cd_tree_tasks"],
)

# Optional configuration, see the application user guide.
app.conf.update(
    result_expires=3600,
)

if __name__ == "__main__":
    app.start()
