import sys

import numpy as np
from celery import Task

sys.path.insert(0, "./cbir/")
sys.path.insert(0, "../cbir/")
from backbone import Backbone
from celery_app import app


class CNNTask(Task):
    abstract = True
    _backbone = None

    @property
    def backbone(self):
        if self._backbone is None:
            backbone = Backbone()
            self._backbone = backbone
        return self._backbone


@app.task(base=CNNTask, bind=True)
def get_features(self, query_img_array):
    return self.backbone.get_features(np.array(query_img_array)).tolist()
