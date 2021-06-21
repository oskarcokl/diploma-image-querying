class CDTree:
    def __init__(self):
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
