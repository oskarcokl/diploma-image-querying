from .celery import app
import sys

sys.path.insert(0, "./cbir/")
from search import search


@app.task
def cbir_query(query_img_path=None, query_img_list=None, cli=False, ):
    result = search(query_img_list=query_img_list, cli=cli)
    return result
