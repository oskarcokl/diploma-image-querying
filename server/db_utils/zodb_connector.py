import BTrees
import transaction
import ZODB
import ZODB.FileStorage
import ZODB.config
from ZODB.POSException import ConnectionStateError


cli = "../db_utils/zodb.zconfig"
server = "./db_utils/zodb.zconfig"

conf = """
%%import relstorage
<zodb main>
<relstorage>
  <postgresql>
    # The dsn is optional, as are each of the parameters in the dsn.
    dsn dbname='%s' user='%s' host='%s' password='%s'
  </postgresql>
</relstorage>
</zodb>
""" % ("image_querying", "postgres", "localhost", "harambe2016!")


class ZODBConnector:
    def __init__(self):
        self.db = None
        self.connection = None
        self.root = None

    def connect(self):
        self.db = ZODB.config.databaseFromString(conf)
        self.connection = self.db.open()
        self.root = self.connection.root

    def disconnect(self):
        self.db.close()

    def save_cd_tree(self, root_node):
        if self.connection is None or self.db is None or self.root is None:
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
