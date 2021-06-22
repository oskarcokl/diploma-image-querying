from gmm import GMM


class CDTree:
    """
    CDTree object:

    Parameters
    ==========

    min_node: int
        Number used to restrict width of tree. If number of feature vectors
        in current node is smaller than min_node then the node will not be split.

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
        gmm_class = GMM()
        stack = []
        root_node = self._generate_root_node(data)
        stack.append(root_node)
        curr_node = stack.pop()

        while curr_node is not None:
            if self.check_stop_conditions(curr_node):
                leaf_node = self.make_node_leaf(curr_node)
            else:
                model = gmm_class.get_optimal_clusters(curr_node_features)
                # Save GMM into curr_node
                for cluster in model.weights:
                    # Add sub nodes to stack
                    pass
            curr_node = stack.pop()

        return root_node

    def _generate_root_node(self, data):

        pass


class _Node:
    """
    Inner node of CDTree:

    Parameters
    ==========

    n_sub_clutsers: int
        Number of sub clusters of this node

    gmm_parameters: list
        Parameters of the gmm model asosiated with this cluster

    sub_nodes: list
        List of pointers to sub nodes (clusters)

    """

    def __init__(self, n_sub_clutsers, gmm_parameters, sub_nodes):
        self.n_sub_clutsers = n_sub_clutsers
        self.gmm_parameters = gmm_parameters
        self.sub_nodes = sub_nodes


class _Leaf:
    """
    Leaf node of CDTree:

    Parameters
    ==========

    n_feature_vectors: int
        Number of feature vectors in this leaf

    ids: list
        List of ids in this leaf.
    """

    def __init__(self, n_feature_vectors, ids):
        self.n_feature_vectors = n_feature_vectors
        self.ids = ids


if __name__ == "__main__":
    print("Testing CDTree class")
