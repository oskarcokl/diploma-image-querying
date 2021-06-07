from .celery import app
import sys

sys.path.append("./scripts/cbir/")
from search import search


@app.task
def cbir_query():
    result = search("../../dataset/OIP-zzy5kEbHuGBWXZ3c-D85aAHaE8.jpeg", False)
    return result
