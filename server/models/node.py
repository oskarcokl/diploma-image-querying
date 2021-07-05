import persistent


class Node(persistent.Persistent):
    """
    Inner node of CDTree:

    Parameters
    ----------

    n_sub_clusters: int
        Number of sub clusters of this node

    gmm_parameters: dict
        Parameters of the gmm model asosiated with this cluster

        Parameters
        ----------
        means: list
            Mean of each gmm cluster.

        covs_array: nd_array
            Covariance matrix of each cluster.

        weights: list
            Weight that each cluster contributes to the gmm.

    sub_nodes: list
        List of pointers to sub nodes (clusters)

    n_feature_vectors: int
        Number of feature vectors in this leaf

    ids: list
        List of ids in this leaf.

    feature_vectors: list
        List of feature vectors.

    img_names: list
        List of img_names. This is only the name of the image. The relative path to where
        it is stored should be added when you wish to retrieve it.

    is_leaf: bool
        Boolean which determines if the node is a leaf or not.

    is_root: bool
        Wheather the node is the rote node of the tree or not.

    layer: int
        Which layer is the node on.
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
        feature_vectors=[],
        img_names=[],
        node_id=-1,
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
        self.feature_vectors = feature_vectors
        self.node_id = node_id
        self.img_names = img_names

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
    node_id: {node_id}
    feature_vectors: {feature_vectors}
    img_names: {img_names}
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
            node_id=self.node_id,
            feature_vectors=self.feature_vectors,
            img_names=self.img_names
        )

    # This function assumes that ids are ordered.
    # Return a list[id:int, feature_vector:[]]
    def get_node_features(self):
        feature_vectors_with_ids = []

        for item in self.data:
            vector_with_id = [item[0], item[1]]
            feature_vectors_with_ids.append(vector_with_id)

        return feature_vectors_with_ids

    def set_feature_vectors(self, feature_vectors):
        self.feature_vectors = feature_vectors
        self._p_changed = True

    def set_ids(self, ids):
        self.ids = ids
        self._p_changed = True

    def set_img_names(self, img_names):
        self.img_names = img_names
        self._p_changed = True

    def set_sub_nodes(self, sub_nodes):
        self.sub_nodes = sub_nodes
        self._p_changed = True

    def set_gmm_parameters(self, gmm_parameters):
        self.gmm_parameters = gmm_parameters
        self._p_changed = True

    def set_covs_array(self, covs_array):
        self.gmm_parameters["covs_array"] = covs_array
        self._p_changed = True

    def set_means(self, means):
        self.gmm_parameters["means"] = means
        self._p_changed = True

    def set_is_leaft(self, is_leaf):
        self.is_leaf = is_leaf
        self._p_changed = True

    def add_id(self, id):
        self.ids.append(id)
        self._p_changed = True

    def add_feature_vector(self, feature_vector):
        self.feature_vectors.append(feature_vector)
        self._p_changed = True

    def make_leaf(self, feature_vectors, img_names):
        self.is_leaf = True
        # Only the leaf nodes should explicitly hold feature vectors,
        # all other nodes should get feature vectors from they're ids.
        self.set_feature_vectors(feature_vectors)
        self.set_img_names(img_names)

    def make_inner_node(self):
        self.is_leaf = False

        # Make features and names empty lists so it doesn't
        # take up space.
        self.set_feature_vectors([])
        self.set_img_names([])
