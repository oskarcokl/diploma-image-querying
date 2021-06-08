from .celery import app
import sys

sys.path.append("./scripts/cbir/")
from search import search


@app.task
def cbir_query(query_img_path=None, query_img_array=None, cli=False):
    print(query_img_array)
    result = search(query_img_array=query_img_array, cli=cli)
    return result
