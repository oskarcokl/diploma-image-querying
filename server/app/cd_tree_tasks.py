import celery
from celery import Task

from app.celery import app
from cbir.search import search
from cbir.add import add
from db_utils.zodb_connector import ZODBConnector


class CDTreeTask(Task):
    abstract = True
    _root_node = None

    @property
    def root_node(self):
        if self._root_node is None:
            zodb_connector = ZODBConnector()
            zodb_connector.connect()
            root_node = zodb_connector.get_root_node()
            self._root_node = root_node
        return self._root_node


@app.task(base=CDTreeTask, bind=True)
def cbir_query(
    self,
    query_img_path=None,
    query_img_list=None,
    cli=False,
    query_features=None,
    n_images=10
):
    root_node = self.root_node
    result = search(
        query_img_list=query_img_list,
        cli=cli,
        query_features=query_features,
        n_images=n_images,
        root_node=root_node
    )
    return result


@app.task(base=CDTreeTask, bind=True)
def index_add(self, decoded_images):
    root_node = self.root_node
    try:
        add(decoded_images, root_node=root_node)
        return True
    except:
        return False
