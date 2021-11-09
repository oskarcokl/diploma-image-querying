import logging

import BTrees
import transaction
import ZODB
import ZODB.FileStorage
import ZODB.config
from ZODB.POSException import ConnectionStateError


from ..models.node import Node

# Configuration for relstorage package. Relstorage allows us to
# use PostgreSQL as storage for ZODB.
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
    """
    ZODBConnector is used to connect to the ZODB database.

    It is also responsible for some basic operations with the
    database. Such as saving the CD-tree and retrieving it.

    Attributes
    ----------
    db : ZODB.DB
        a ZODB object databse object.
    connection : IConnection
        a IConnection which has a connection to the database. 
    root: Root object of database
        top level object in database. Root node of CD-tree is
        a sub object of the root.
    """

    def __init__(self):
        self.db = None
        self.connection = None
        self.root = None

    def connect(self):
        logging.info("Connection to ZODB.")
        self.db = ZODB.config.databaseFromString(conf)
        self.connection = self.db.open()
        self.root = self.connection.root

    def disconnect(self):
        self.db.close()

    def save_cd_tree(self, root_node: Node):
        if self.connection is None or self.db is None or self.root is None:
            raise ConnectionStateError("ERROR: Not connected to ZODB database")

        self.root.cd_tree = BTrees.OOBTree.BTree()
        self.root.cd_tree["root_node"] = root_node
        logging.info("Saving cd tree")
        transaction.commit()
        logging.info("CD tree saved")

    def get_root_node(self) -> Node:
        if self.connection is None or self.db is None or self.root is None:
            raise ConnectionStateError("ERROR: Not connected to ZODB database")
        return self.root.cd_tree["root_node"]
