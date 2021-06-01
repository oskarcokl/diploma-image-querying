import sys

sys.path.append("../")

import numpy as np
from sklearn.neighbors import NearestNeighbors
import psycopg2
from db_connector import DbConnector


class Searcher:
    def search(self, query_features, n_neigbhours):
        results = {}
        feature_list = []

        connector = DbConnector()
        print(dir(connector))


if __name__ == "__main__":
    Searcher().search("bruh", "cringe")
