import os
import numpy as np
from sklearn.neighbors import NearestNeighbors
import psycopg2
import db_connector


class Searcher:
    def __init__(self, index_path):
        self.index_path = index_path

    def search(self, query_features, n_neigbhours):
        results = {}
        feature_list = []

        connector = db_connector()
