import sys
from sklearn.neighbors import NearestNeighbors
import ZODB
import ZODB.FileStorage


sys.path.append("../")
sys.path.append("./scripts/")
from db_connector import DbConnector


class Searcher:
    def _get_root_node(self):
        storage = ZODB.FileStorage.FileStorage("cd_tree.fs")
        db = ZODB.DB(storage)
        connection = db.open()
        root = connection.root

        return root.cd_tree["root_node"]

    def search(self, query_features, n_similar_images):
        root_node = self._get_root_node()
        print(root_node.sub_nodes[1])
        pass


if __name__ == "__main__":
    Searcher().search("bruh", "cringe")
