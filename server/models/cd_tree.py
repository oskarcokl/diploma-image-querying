from os import kill
from gmm import GMM
import numpy as np
import persistent


class CDTree(persistent.Persistent):
    """
    CDTree object:

    Parameters
    ----------

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
        self.node_id = 1

    """
    Function that initializes the CDTree and returns it.

    Parameters
    ----------

    data: [id: int, img_src: string, feature_vector: [int]]
    """

    def init_cd_tree(
        self,
        data,
        tolerance=0.001,
        n_iters=1000,
    ):
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
                model = gmm.get_optimal_clusters(
                    curr_node_feature_array, 1, 2, tolerance, n_iters
                )
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

                sub_nodes = self._create_sub_nodes(
                    vectors_with_clusters,
                    curr_node.ids,
                    curr_node.layer + 1,
                    len(model.weights),
                )

                for sub_node in sub_nodes:
                    stack.append(sub_node)

                curr_node.sub_nodes = sub_nodes
                curr_node.n_sub_clusters = len(sub_nodes)

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
            node_id=self.node_id,
        )
        return root_node

    # (2π)^(−d/2) * |Σi|^(−1/2) * exp(−12(X−μi)TΣ−1i(X−μi)).
    # Function calculates the above equation
    def _calculate_cpd(self, feature_vector, mean, cov_array):
        d = len(feature_vector)
        a = (2 * np.pi) ** (d / 2)
        b = np.linalg.det(cov_array) ** -0.5
        c = np.exp(
            -0.5
            * np.matmul(
                np.matmul(
                    np.transpose(feature_vector - mean), np.linalg.inv(cov_array)
                ),
                (feature_vector - mean),
            )
        )
        cpd = a * b * c
        return cpd

    # Function calculates the new mean and cov matrix for a node.
    # These values need to be updated when we want to insert a new,
    # data point. M is the number of data points in the node we are updating
    def _calculate_mean_and_cov(self, m, feature_vector, mean, cov_array):
        new_mean = ((m / (m + 1)) * mean) + ((1 / (m + 1)) * feature_vector)
        new_cov_array = (((m - 1) / m) * cov_array) + (
            (1 / (m + 1))
            * np.matmul((feature_vector - mean), (feature_vector - mean).T)
        )
        return new_mean, new_cov_array

    # Algorithm finds leaf node of query_feature_vector. It finds
    # the leaf by calculating cpd's for each subnode and choosing
    # the subnode with the highest cpd.
    def _find_leaf_node(self, id, query_feature_vector, root_node):

        # TODO update parameters of root node.
        curr_node = root_node
        while not curr_node.is_leaf:
            max_cpd = -1
            max_node_index = -1
            max_node_mean
            max_node_cov_array
            for i in range(curr_node.n_sub_clusters):
                mean = curr_node.gmm_parameters["means"][i]
                cov_array = curr_node.gmm_parameters["covs_array"][i]
                cpd = self._calculate_cpd(query_feature_vector, mean, cov_array)
                if max_cpd < cpd:
                    max_cpd = cpd
                    max_node_index = i
                    max_node_mean = mean
                    max_node_cov_array = cov_array

            sub_node = curr_node.sub_nodes[max_node_index]
            # For now I will be adding feature vectors and ids to
            # all of the nodes in the path but this is generally not
            # needed.
            sub_node.ids.append(id)
            sub_node.feature_vectors.append(query_feature_vector)
            sub_node.n_feature_vectors += 1

            # Calculate mean and cov and update them for the node
            # with the max cpd.
            (
                curr_node.gmm_parameters["means"][max_node_index],
                curr_node.gmm_parameters["covs_array"][max_node_index],
            ) = self._calculate_mean_and_cov(
                sub_node.n_feature_vectors,
                query_feature_vector,
                max_node_mean,
                max_node_cov_array,
            )

            curr_node = sub_node

        curr_node.ids.append(id)
        curr_node.feature_vectors.append(query_feature_vector)
        curr_node.n_feature_vectors += 1

        return curr_node

    def add_feature_vector(self, id, query_feature_vector, root_node):
        # Used to determine if leaf need to be split with new
        # data insertion. Could be set by user.
        gama = 0.1

        curr_node = self._find_leaf_node(id, query_feature_vector, root_node)

        # Split the leaf node into to nodes if parent n features * gama is
        # smaller than then feature of the leaf node.

        n_feature_vectors_parent = curr_node.n_feature_vectors

        # TODO Can be refractored to look a lot better.
        if curr_node.n_feature_vectors > gama * n_feature_vectors_parent:
            curr_node.is_leaf = False
            gmm = GMM()
            model = gmm.gmm_clustering(np.array(curr_node.feature_vectors), 2)
            curr_node.gmm_parameters = {
                "covs_array": model.covs_array,
                "means": model.means,
                "weights": model.weights,
            }

            # Might turned this into a function since the same thing
            # happens in init_cd_tree().

            vectors_with_clusters = self._asign_vectors_to_clusters(
                curr_node.ids, curr_node.feature_vectors, model.resp_array
            )

            sub_nodes = self._create_sub_nodes(
                vectors_with_clusters, curr_node.ids, curr_node.layer + 1, 2
            )

            curr_node.sub_nodes = sub_nodes
            curr_node.n_sub_clusters = 2

        return root_node

    def _create_sub_nodes(
        self, feature_vectors_with_clusters, ids, sub_node_layer, n_clusters
    ):
        sub_nodes = []

        # Outer loop loops throught all of the clusters and creates
        # a new node for each one.
        for index in range(n_clusters):
            self.node_id += 1
            sub_node = _Node(layer=sub_node_layer, node_id=self.node_id)

            feature_vectors = []
            ids = []

            # Iterates over feature vectors and pust the ones asigned to current
            # cluster into a list which is then added to the new node.
            for item in feature_vectors_with_clusters:
                if item[2] == index:
                    feature_vectors.append(item[1])
                    ids.append(item[0])

            sub_node.ids = ids
            sub_node.feature_vectors = feature_vectors
            sub_node.n_feature_vectors = len(ids)
            sub_nodes.append(sub_node)

        return sub_nodes

    def _calculate_eucledian_distance(self, query_feature_vector, feature_vector):
        result_array = np.linalg.norm(
            np.array(query_feature_vector) - np.array(feature_vector)
        )
        return result_array.tolist()

    def _rank_images(self, query_feature_vector, similar_images):
        for i in range(len(similar_images)):
            d = self._calculate_eucledian_distance(
                query_feature_vector, similar_images[i][1]
            )
            similar_images[i].append(d)

        ranked_similar_images = sorted(similar_images, reverse=True, key=lambda x: x[2])
        return ranked_similar_images

    # K: int. Number of similar images to return
    def find_similar_images(self, root_node, query_feature_vector, q, k):
        stack = []
        stack.append(root_node)
        n_data_points = 0
        similar_data_points = []
        while n_data_points < k:
            curr_node = stack.pop()
            if not curr_node.is_leaf:
                cpds = []
                for i in range(curr_node.n_sub_clusters):
                    mean = curr_node.gmm_parameters["means"][i]
                    cov_array = curr_node.gmm_parameters["covs_array"][i]
                    cpd = self._calculate_cpd(query_feature_vector, mean, cov_array)
                    # Save index of each sub node alongside the cpd. this is done
                    # so that sub nodes can be added in decreasing relevance based
                    # on cpd.
                    cpds.append([i, cpd])

                sorted_cpds = sorted(cpds, key=lambda x: x[1], reverse=True)

                for item in sorted_cpds:
                    stack.append(curr_node.sub_nodes[item[0]])
            else:
                for i in range(curr_node.n_feature_vectors):
                    similar_data_points.append(
                        [curr_node.ids[i], curr_node.feature_vectors[i]]
                    )
                n_data_points = len(similar_data_points)

        return self._rank_images(query_feature_vector, similar_data_points)


class _Node(persistent.Persistent):
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

    def set_feature_vectors(self, feature_vectors):
        self.feature_vectors = feature_vectors
        self._p_changed = True

    def set_ids(self, ids):
        self.ids = ids
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

    def add_id(self, id):
        self.ids.append(id)
        self._p_changed = True

    def add_feature_vector(self, feature_vector):
        self.feature_vectors.append(feature_vector)
        self._p_changed = True


# Used for quick testing during developement.
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
