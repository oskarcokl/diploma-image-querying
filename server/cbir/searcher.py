import sys
import pickle

import ZODB
import ZODB.FileStorage
from codetiming import Timer
from sklearn.neighbors import NearestNeighbors


sys.path.insert(0, "../")
sys.path.append("./scripts/")
from models import cd_tree


class Searcher:
    def _get_root_node(self):
        storage = ZODB.FileStorage.FileStorage("./cbir/cd_tree.fs")
        db = ZODB.DB(storage)
        connection = db.open()
        root = connection.root

        return root.cd_tree["root_node"]

    def search(self, query_features, n_similar_images):
        search_time = Timer(name="Search", logger=None)
        search_time.start()
        root_node = self._get_root_node()
        result_images = cd_tree.find_similar_images(
            root_node=root_node,
            query_feature_vector=query_features,
            n_similar_images=n_similar_images)

        img_names = []

        for i in range(n_similar_images):
            img_names.append(result_images[i][2])

        elapsed_time = search_time.stop()
        return img_names, elapsed_time

    def search_force(self, query_features, feature_vectors, img_names, n_similar_images):
        search_time = Timer(name="Search", logger=None)
        search_time.start()

        result_img_names = []

        neighbors = NearestNeighbors(n_neighbors=1)
        neighbors.fit(feature_vectors)
        _, indexes = neighbors.kneighbors(
            X=query_features.reshape(1, -1), n_neighbors=n_similar_images)

        for index in indexes[0]:
            result_img_names.append(img_names[index])

        elapsed_time = search_time.stop()
        return result_img_names, elapsed_time


if __name__ == "__main__":
    query_feature_vector = []

    with open("query_features", "rb") as f:
        query_feature_vector = pickle.load(f)

    Searcher().search(n_similar_images=10, query_features=query_feature_vector)
