from celery import Celery
from kombu.common import Broadcast
from app import celeryconfig

app = Celery("app")

app.config_from_object(celeryconfig)


app.conf.task_queues = (Broadcast('broadcast_tasks'),)
app.conf.task_routes = {
    "cd_tree_tasks.reload_cd_tree": {
        "queue": "broadcast_tasks",
        "exchange": "broadcast_tasks"
    }
}

if __name__ == "__main__":
    app.start()
