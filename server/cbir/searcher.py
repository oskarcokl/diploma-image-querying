from re import S
import sys
import pickle

from sklearn.neighbors import NearestNeighbors
import ZODB
import ZODB.FileStorage


sys.path.insert(0, "../")
sys.path.append("./scripts/")
from db_connector import DbConnector
from models import cd_tree

# Testing vars


class Searcher:
    def _get_root_node(self):
        storage = ZODB.FileStorage.FileStorage("cd_tree.fs")
        db = ZODB.DB(storage)
        connection = db.open()
        root = connection.root

        return root.cd_tree["root_node"]

    def search(self, query_features, n_similar_images):
        root_node = self._get_root_node()
        result_images = cd_tree.find_similar_images(
            root_node=root_node,
            query_feature_vector=query_features,
            n_similar_images=n_similar_images)

        img_names = []

        for i in range(n_similar_images):
            img_names.append(result_images[i][2])

        return img_names


if __name__ == "__main__":
    query_feature_vector = []

    with open("query_features", "rb") as f:
        query_feature_vector = pickle.load(f)

    Searcher().search(n_similar_images=10, query_features=query_feature_vector)
