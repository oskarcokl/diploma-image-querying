from re import A
import sys

sys.path.append("../")

import numpy as np
from sklearn.neighbors import NearestNeighbors
from db_connector import DbConnector


class Searcher:
    def search(self, query_features, n_neigbhours):
        results = {}
        feature_list = []

        connector = DbConnector()
        connector.cursor.execute("SELECT * FROM cbir_index")
        print("Number of indexed images: ", connector.cursor.rowcount)
        cbir_index_features = connector.cursor.fetchall()
        cbir_index_img_paths = []
        features_array_list = []

        for cbir_index_feature_list in cbir_index_features:
            cbir_index_img_paths.append(cbir_index_feature_list[1])
            feature_array = np.array(cbir_index_feature_list[2])
            features_array_list.append(feature_array)

        features_array = np.array(features_array_list)

        neighbor_model = NearestNeighbors(n_neighbors=n_neigbhours)
        neighbor_model.fit(features_array)

        dist, results = neighbor_model.kneighbors(query_features)

        connector.close()
        img_paths = []

        for result in results[0]:
            img_paths.append(cbir_index_img_paths[result])

        return dist[0], img_paths


if __name__ == "__main__":
    Searcher().search("bruh", "cringe")
