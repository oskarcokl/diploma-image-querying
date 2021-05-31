import os
import numpy as np
from sklearn.neighbors import NearestNeighbors


class Searcher:
    def __init__(self, index_path):
        self.index_path = index_path

    def search(self, query_features, neigbhours):
        results = {}
        feature_list = []
        img_ids_all = []
