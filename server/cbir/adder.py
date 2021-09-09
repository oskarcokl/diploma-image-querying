import sys


sys.path.insert(0, "../")
sys.path.insert(0, "./")


from models import cd_tree
from db_utils import table_operations


class Adder:
    def add_img_to_db(self, tuple_list):
        ids = table_operations.insert_many_tuples(tuple_list)
        return ids

    def add_to_cd_tree(self, id, img_features, img_name, root_node):
        new_root_node = cd_tree.add_to_cd_tree(
            id, img_features, img_name, root_node)
        return new_root_node
