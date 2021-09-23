import numpy as np
from celery import Task

from cbir.backbone import Backbone
from db_utils.table_operations import get_feature_vectos_all
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
        if self._feature_vectors is None:
            self._feature_vectors = get_feature_vectos_all()
        return self._feature_vectors


@app.task(base=CNNTask, bind=True)
def get_features(self, query_img_array):
    return self.backbone.get_features(np.array(query_img_array)).tolist()
