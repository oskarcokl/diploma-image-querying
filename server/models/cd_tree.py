from os import kill
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
        gmm = GMM()
        stack = []
        root_node = self._generate_root_node(data)
        stack.append(root_node)
        curr_node = stack.pop()

        print(curr_node)

        while curr_node is not None:
            if self._check_stop_conditions(curr_node):
                curr_node.is_leaf = True
            else:
                curr_node_features = self._get_node_features(curr_node, data)
                model = gmm.get_optimal_clusters(curr_node_features)
                # Save GMM into curr_node
                for cluster in model.weights:
                    # Add sub nodes to stack
                    pass
            curr_node = stack.pop()

        return root_node

    def _check_stop_conditions(self, node):
        if len(node._gmm_parameters["weights"]) == 1:
            return False
        elif node._n_feature_vectors < self.min_node:
            return False
        elif node._layer >= self.l_max:
            return False

        return True

    def _generate_sub_node(self):
        pass

    def _generate_root_node(self, data):
        ids = []
        for item in data:
            ids.append(item[0])
        root_node = _Node(
            n_feature_vectors=len(data),
            is_leaf=False,
            ids=ids,
            is_root=True,
            layer=0,
            data=data,
        )
        return root_node


class _Node:
    """
    Inner node of CDTree:

    Parameters
    ==========

    n_sub_clusters: int
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

    layer: int
        Which layer is the node on.

    data: []
        Same as data that gets based into CDTree.init_cd_tree()
        but only keeping the relevant parts for the node.

    """

    def __init__(
        self,
        is_leaf=False,
        n_feature_vectors=-1,
        ids=[],
        n_sub_clusters=-1,
        gmm_parameters={},
        sub_nodes=[],
        is_root=False,
        layer=-1,
        data=[],
    ):
        self.is_leaf = is_leaf
        self.n_feature_vectors = n_feature_vectors
        self.ids = ids
        self.is_root = is_root
        self.n_sub_clusters = n_sub_clusters
        self.gmm_parameters = gmm_parameters
        self.sub_nodes = sub_nodes
        self.layer = layer
        self.data = data

    def __str__(self):
        return """
    Information about nodes paramteres
    ==================================
    is_leaf: {is_leaf}
    n_feature_vectors: {n_feature_vectors}
    ids: {ids}
    n_sub_clutsers: {n_sub_clusters}
    gmm_parameters: {gmm_parameters}
    sub_nodes: {sub_nodes}
    is_root: {is_root}
    layer: {layer}
    ==================================
        """.format(
            is_leaf=self.is_leaf,
            n_feature_vectors=self.n_feature_vectors,
            ids=self.ids,
            is_root=self.is_root,
            n_sub_clusters=self.n_sub_clusters,
            gmm_parameters=self.gmm_parameters,
            sub_nodes=self.sub_nodes,
            layer=self.layer,
        )

    # This function assumes that ids are ordered.
    # Return a list[id:int, feature_vector:[]]
    def _get_node_features(self):
        feature_vectors_with_ids = []

        for item in self.data:
            vector_with_id = [item[0], item[1]]
            feature_vectors_with_ids.append(vector_with_id)

        return feature_vectors_with_ids


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
