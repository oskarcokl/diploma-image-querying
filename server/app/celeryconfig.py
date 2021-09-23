broker_url = "amqp://"
result_backend = "rpc://"
imports = ["app.cd_tree_tasks", "app.cnn_tasks"]

task_routes = {
    "app.cd_tree_tasks": "cd_tree",
    "app.cnn": "cnn"
}
result_expires = 60
