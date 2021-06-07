from .celery import app


@app.task
def cbir_query():
    return "Your images."
