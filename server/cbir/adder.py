import sys

import ZODB
import ZODB.FileStorage.FileStorage

sys.path.insert(0, "../")
sys.path.append("./scripts/")
from models import cd_tree


class Adder:
    def _get_root_node(self):
        storage = ZODB.FileStorage("./cbir/cd_tree.fs")
        db = ZODB.DB(storage)
        connection = db.open()
        root = connection.root

        return root.cd_tree["root_node"]

    def add_img_to_db(self):
        pass

    def add_to_cd_tree(self, id, img_features):
        root_node = self._get_root_node()
        node = cd_tree.add_to_cd_tree(id, img_features, root_node)
        print(node)
