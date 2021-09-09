import sys
import pickle

import ZODB
import ZODB.FileStorage
from codetiming import Timer
from sklearn.neighbors import NearestNeighbors
import numpy as np


sys.path.insert(0, "../")
sys.path.append("./scripts/")
from models import cd_tree


cli = "../cbir/cd_tree.fs"
server = "./cbir/cd_tree.fs"


class Searcher:
    root = None

    def _get_root_node(self):
        if self.root:
            return self.root.cd_tree["root_node"]
        else:
            self.storage = ZODB.FileStorage.FileStorage(server)
            self.db = ZODB.DB(self.storage)
            self.connection = self.db.open()
            self.root = self.connection.root

            return self.root.cd_tree["root_node"]

    def _close_connection(self):
        print("Closing connection")
        self.db.close()

    def search(self, query_features, n_similar_images):
        root_node = self._get_root_node()
        search_time = Timer(name="Search", logger=None)
        search_time.start()
        result_images = cd_tree.find_similar_images(
            root_node=root_node,
            query_feature_vector=query_features,
            n_similar_images=n_similar_images)

        elapsed_time = search_time.stop()
        self._close_connection()
        img_names = []

        for i in range(n_similar_images):
            img_names.append(result_images[i][2])

        return img_names, elapsed_time

    def search_force(self, query_features, feature_vectors, img_names, n_similar_images):

        search_time = Timer(name="Search", logger=None)
        search_time.start()

        result_img_names = []

        #neighbors = NearestNeighbors(n_neighbors=1, algorithm="brute")

        distances = self._compute_distances(
            query_features, feature_vectors, img_names)
        # neighbors.fit(feature_vectors)
        # _, indexes = neighbors.kneighbors(
        #     X=query_features.reshape(1, -1), n_neighbors=n_similar_images)

        # for index in indexes[0]:
        #     result_img_names.append(img_names[index])

        ranked_distances = sorted(distances, reverse=True, key=lambda x: x[0])

        for i in range(n_similar_images):
            result_img_names.append(ranked_distances[i][1])

        elapsed_time = search_time.stop()
        return result_img_names, elapsed_time

    def _compute_distances(self, query_features, feature_vectors, img_names):
        results = []
        for index, feature_vector in enumerate(feature_vectors):
            d = self._compute_cos_similarity(query_features, feature_vector)
            results.append((d, img_names[index]))

        return results

    def _compute_cos_similarity(self, x, y):
        dot_product = np.dot(x, y)
        magnitude_q = np.linalg.norm(x)
        magnitude_f = np.linalg.norm(y)
        result = dot_product / (magnitude_f * magnitude_q)
        return result


if __name__ == "__main__":
    query_feature_vector = []

    with open("query_features", "rb") as f:
        query_feature_vector = pickle.load(f)

    Searcher().search(n_similar_images=10, query_features=query_feature_vector)
