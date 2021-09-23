import logging

from celery import Task
from kombu.entity import Queue

from app.celery import app
from cbir.search import search
from cbir.add import add
from db_utils.zodb_connector import ZODBConnector


class CDTreeTask(Task):
    abstract = True
    _root_node = None
    _zodb_connector = None

    @property
    def root_node(self):
        if self._root_node is None:
            self.zodb_connector = ZODBConnector()
            self.zodb_connector.connect()
            root_node = self.zodb_connector.get_root_node()
            self._root_node = root_node
        return self._root_node

    def reload_cd_tree(self):
        if self._root_node is None:
            self.zodb_connector = ZODBConnector()
            self.zodb_connector.connect()
        else:
            self.zodb_connector.disconnect()
            self.zodb_connector.connect()

        root_node = self.zodb_connector.get_root_node()
        self._root_node = root_node


@app.task(base=CDTreeTask, bind=True)
def cbir_query(
    self,
    query_img_path=None,
    query_img_list=None,
    cli=False,
    query_features=None,
    n_images=10,
    feature_vectors=None
):
    root_node = self.root_node
    result = search(
        query_img_list=query_img_list,
        cli=cli,
        query_features=query_features,
        n_images=n_images,
        feature_vectors=feature_vectors,
        root_node=root_node
    )
    return result


@app.task(base=CDTreeTask, bind=True)
def index_add(self, decoded_images):
    root_node = self.root_node
    reload_cd_tree_task.apply_async(queue="broadcast")
    try:
        add(decoded_images, root_node=root_node)
        return True
    except Exception as e:
        print(e)
        return False


@app.task(base=CDTreeTask, bind=True)
def reload_cd_tree_task(self):
    try:
        print("Reloading CD-tree.")
        self.reload_cd_tree()
    except Exception as e:
        logging.exception("Exception occured while reloading CD-tree.", e)
