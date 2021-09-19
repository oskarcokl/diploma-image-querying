import sys

from codetiming import Timer
import numpy as np

sys.path.insert(0, "../")
sys.path.append("./scripts/")
from models import cd_tree
from db_utils.zodb_connector import ZODBConnector


class Searcher:
    def search(self, query_features, n_similar_images, root_node=None):
        if root_node is None:
            print("Root node is None")
            zodb_connector = ZODBConnector()
            zodb_connector.connect()
            root_node = zodb_connector.get_root_node()

        search_time = Timer(name="Search", logger=None)
        search_time.start()
        result_images = cd_tree.find_similar_images(
            root_node=root_node,
            query_feature_vector=query_features,
            n_similar_images=n_similar_images)

        elapsed_time = search_time.stop()
        img_names = []

        for i in range(n_similar_images):
            img_names.append(result_images[i][1])

        return img_names, elapsed_time

    def search_force(self, query_features, feature_vectors, img_names, n_similar_images):

        search_time = Timer(name="Search", logger=None)
        search_time.start()

        result_img_names = []

        distances = self._compute_distances(
            query_features, feature_vectors, img_names)

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
