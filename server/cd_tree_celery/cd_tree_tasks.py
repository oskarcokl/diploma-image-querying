from .celery import app
import sys

sys.path.insert(0, "./cbir/")
from search import search
from add import add


@app.task
def cbir_query(query_img_path=None, query_img_list=None, cli=False, query_features=None, n_images=10):
    result = search(query_img_list=query_img_list,
                    cli=cli, query_features=query_features, n_images=n_images)
    return result


@app.task
def index_add(decoded_images):
    try:
        add(decoded_images)
        return True
    except:
        return False
