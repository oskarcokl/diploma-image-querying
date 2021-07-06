import ZODB
import ZODB.FileStorage.FileStorage


class Adder:
    def _get_root_node(self):
        storage = ZODB.FileStorage("./cbir/cd_tree.fs")
        db = ZODB.DB(storage)
        connection = db.open()
        root = connection.root

        return root.cd_tree["root_node"]
