import BTrees
import transaction
import ZODB
import ZODB.FileStorage
import ZODB.config
from ZODB.POSException import ConnectionStateError


cli = "../db_utils/zodb.zconfig"
server = "./db_utils/zodb.zconfig"


class ZODBConnector:
    def __init__(self):
        self.db = None
        self.connection = None
        self.root = None

    def connect(self):
        self.db = ZODB.config.databaseFromURL(cli)
        self.connection = self.db.open()
        self.root = self.connection.root

    def disconnect(self):
        self.db.close()

    def save_cd_tree(self, root_node):
        if self.storage is None or self.connection is None or self.db is None or self.root is None:
            raise ConnectionStateError("ERROR: Not connected to ZODB database")

        self.root.cd_tree = BTrees.OOBTree.BTree()
        self.root.cd_tree["root_node"] = root_node
        print("Saving cd tree")
        transaction.commit()
        print("CD tree saved")

    def get_root_node(self):
        if self.connection is None or self.db is None or self.root is None:
            raise ConnectionStateError("ERROR: Not connected to ZODB database")
        return self.root.cd_tree["root_node"]
