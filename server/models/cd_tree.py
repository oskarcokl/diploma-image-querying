from gmm import GMM
import numpy as np


class CDTree:
    """
    CDTree object:

    Parameters
    ==========

    min_node: int
        Number used to restrict width of tree. If number of feature vectors
        in current number of feature vectors is smaller than min_node then
        the node will not be split.

    l_max: int
        Number used to restrict depth of tree. If layer of sub node with be greater
        than l_max the node will not be split
    """

    def __init__(self, min_node, l_max):
        self.min_node = min_node
        self.l_max = l_max

    """
    Function that initializes the CDTree and returns it.

    Parameters
    ==========

    data: [id: int, img_src: string, feature_vector: [int]]
    """

    def init_cd_tree(self, data):
        stack = []
        root_node = self._generate_root_node(data)
        stack.append(root_node)
        curr_node = stack.pop()

        # while curr_node is not None:
        #     if self.check_stop_conditions(curr_node):
        #         leaf_node = self.make_node_leaf(curr_node)
        #     else:
        #         model = gmm_class.get_optimal_clusters(curr_node_features)
        #         # Save GMM into curr_node
        #         for cluster in model.weights:
        #             # Add sub nodes to stack
        #             pass
        #     curr_node = stack.pop()

        # return root_node
        pass

    def _generate_root_node(self, data):
        ids = []
        for item in data:
            ids.append(item[0])
        root_node = _Node(
            n_feature_vectors=len(data), is_leaf=False, ids=ids, is_root=True
        )
        return root_node


class _Node:
    """
    Inner node of CDTree:

    Parameters
    ==========

    n_sub_clutsers: int
        Number of sub clusters of this node

    gmm_parameters: dict
        Parameters of the gmm model asosiated with this cluster

    sub_nodes: list
        List of pointers to sub nodes (clusters)

    n_feature_vectors: int
        Number of feature vectors in this leaf

    ids: list
        List of ids in this leaf.

    is_leaf: bool
        Boolean which determines if the node is a leaf or not.

    is_root: bool
        Wheather the node is the rote node of the tree or not.

    """

    def __init__(
        self,
        is_leaf=False,
        n_feature_vectors=0,
        ids=[],
        n_sub_clutsers=0,
        gmm_parameters={},
        sub_nodes=[],
        is_root=False,
    ):
        if is_leaf:
            self._is_leaf = is_leaf
            self._n_feature_vectors = n_feature_vectors
            self._ids = ids
            self._is_root = is_root
        else:
            self._is_leaf = is_leaf
            self._n_sub_clutsers = n_sub_clutsers
            self._gmm_parameters = gmm_parameters
            self._sub_nodes = sub_nodes
            self._is_root = is_root


class _Leaf:
    """
    Leaf node of CDTree:

    Parameters
    ==========


    """

    def __init__(self, n_feature_vectors, ids):
        self.n_feature_vectors = n_feature_vectors
        self.ids = ids


if __name__ == "__main__":
    print("Testing CDTree class")

    # Test data
    data_list = []

    for id in range(1, 21):
        rand_feature_vector = np.random.rand(40)
        temp_list = [id, rand_feature_vector]
        data_list.append(temp_list)

    cd_tree = CDTree(min_node=20, l_max=4)
    cd_tree.init_cd_tree(data_list)
