import sys

import BTrees
import transaction
import ZODB
import ZODB.FileStorage
from ZODB.POSException import ConnectionStateError

sys.path.insert(0, "../")
sys.path.insert(0, "./")


from models import cd_tree
from db_utils import table_operations


class Adder:
    def add_img_to_db(self, img_features, img_name):
        id = table_operations.insert_tuple((img_name, img_features))
        return id

    def add_to_cd_tree(self, id, img_features, img_name):
        root_node = self._get_root_node()
        new_root_node = cd_tree.add_to_cd_tree(
            id, img_features, img_name, root_node)
        return new_root_node


class ZODBConnector:
    def connect(self, file_name):
        self.storage = ZODB.FileStorage.FileStorage(file_name)
        self.db = ZODB.DB(self.storage)
        self.connection = self.db.open()
        self.root = self.connection.root

    def save_cd_tree(self, root_node):
        if self.storage is None or self.connection is None or self.db is None or self.root is None:
            raise ConnectionStateError("ERROR: Not connected to ZODB database")

        self.root.cd_tree = BTrees.OOBTree.BTree()
        self.root.cd_tree["root_node"] = root_node
        print("Saving cd tree")
        transaction.commit()
        print("CD tree saved")

    def get_root_node(self):
        if self.storage is None or self.connection is None or self.db is None or self.root is None:
            raise ConnectionStateError("ERROR: Not connected to ZODB database")
        return self.root.cd_tree["root_node"]
