import sys
import os

import ZODB
import ZODB.FileStorage

sys.path.insert(0, "../")
sys.path.insert(0, "./")


from models import cd_tree
from db_utils import table_operations


class Adder:
    def _get_root_node(self):
        storage = ZODB.FileStorage.FileStorage("./cd_tree.fs")
        db = ZODB.DB(storage)
        connection = db.open()
        root = connection.root

        return root.cd_tree["root_node"]

    def add_img_to_db(self, img_features, img_name):
        id = table_operations.insert_tuple((img_name, img_features))
        return id

    def add_to_cd_tree(self, id, img_features, img_name):
        root_node = self._get_root_node()
        node = cd_tree.add_to_cd_tree(id, img_features, img_name, root_node)
        return node
