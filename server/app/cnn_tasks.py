from logging import exception
import numpy as np
from celery import Task

from cbir.backbone import Backbone
from db_utils.table_operations import get_feature_vectors_all
from app.celery import app


class CNNTask(Task):
    abstract = True
    _backbone = None
    _feature_vectors = []

    @property
    def backbone(self):
        if self._backbone is None:
            self._backbone = Backbone()
        return self._backbone

    @property
    def feature_vectors(self):
        if not self._feature_vectors:
            self._feature_vectors = get_feature_vectors_all().tolist()
        return self._feature_vectors


@app.task(base=CNNTask, bind=True)
def get_features(self, query_img_array):
    return self.backbone.get_features(np.array(query_img_array)).tolist()


@app.task(base=CNNTask, bind=True)
def get_feature_vectors_all_task(self):
    try:
        return self.feature_vectors
    except Exception as e:
        print(e)
