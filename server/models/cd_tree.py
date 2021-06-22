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

        while curr_node is not None:
            if self._check_stop_conditions(curr_node):
                curr_node.is_leaf = True
            else:
                curr_node_feature_array = np.array(curr_node.feature_vectors)
                model = gmm.get_optimal_clusters(curr_node_feature_array, 1, 2)
                node_gmm_parameters = {
                    "covs_array": model.covs_array,
                    "means": model.means,
                    "weights": model.weights,
                }
                # Save GMM parameters into curr_node
                curr_node.gmm_parameters = node_gmm_parameters
                vectors_with_clusters = self._asign_vectors_to_clusters(
                    curr_node.ids, curr_node.feature_vectors, model.resp_array
                )

                new_nodes = []
                node_id = 1
                # Outer loop loops throught all of the clusters and creates
                # a new node for each one.
                for index in range(len(model.weights)):
                    new_node = _Node(layer=curr_node.layer + 1, node_id=node_id)
                    node_id += 1

                    feature_vectors = []
                    ids = []

                    # Iterates over feature vectors and pust the ones asigned to current
                    # cluster into a list which is then added to the new node.
                    for item in vectors_with_clusters:
                        if item[2] == index:
                            feature_vectors.append(item[1])
                            ids.append(item[0])

                    new_node.ids = ids
                    new_node.feature_vectors = feature_vectors
                    new_node.n_feature_vectors = len(ids)
                    new_nodes.append(new_node)
                    stack.append(new_node)

                curr_node.sub_nodes = new_nodes
                curr_node.n_sub_clusters = len(new_nodes)

            if stack:
                curr_node = stack.pop()
            else:
                return root_node

    def _asign_vectors_to_clusters(self, ids, feature_vectors, resp_array):
        new_data = []
        for i in range(len(ids)):
            cluster = self._get_cluster_of_index(resp_array, i)
            new_data_item = [ids[i], feature_vectors[i], cluster]
            new_data.append(new_data_item)

        return new_data

    def _get_cluster_of_index(self, resp_array, index):
        n_clusters = resp_array.shape[1]
        # Probability (responsibility) can never be smaller than 0.
        max_resp = -1
        cluster = -1

        for i in range(n_clusters):
            if resp_array[index][i] > max_resp:
                max_resp = resp_array[index][i]
                cluster = i

        return cluster

    # If stop conditions have been met the functino returns true.
    # Else it retusn false.
    def _check_stop_conditions(self, node):
        if node.gmm_parameters and len(node.gmm_parameters["weights"]) == 1:
            return True
        elif node.n_feature_vectors < self.min_node:
            return True
        elif node.layer >= self.l_max:
            return True

        return False

    def _generate_sub_node(self):
        pass

    def _generate_root_node(self, data):
        ids = []
        feature_vectors = []
        for item in data:
            ids.append(item[0])
            feature_vectors.append(item[2])

        root_node = _Node(
            n_feature_vectors=len(data),
            is_leaf=False,
            ids=ids,
            is_root=True,
            layer=0,
            feature_vectors=feature_vectors,
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
        feature_vectors=[],
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
        )

    # This function assumes that ids are ordered.
    # Return a list[id:int, feature_vector:[]]
    def get_node_features(self):
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
        temp_list = [id, "img_src", rand_feature_vector]
        data_list.append(temp_list)

    cd_tree = CDTree(min_node=20, l_max=4)
    root = cd_tree.init_cd_tree(data_list)

    print(root)
    print(root.sub_nodes[0])
    print(root.sub_nodes[1])
