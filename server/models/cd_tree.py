import numpy as np
from numpy.core.numeric import Inf
import persistent
from sklearn import mixture
from scipy import stats

from .gmm import GMM
from node import Node

# TODO refactor to not be a class and rather a module.


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
        min_clusters,
        max_clusters,
        tolerance=0.001,
        n_iters=1000,
    ):
        stack = []
        root_node = self._generate_root_node(data)
        stack.append(root_node)
        curr_node = stack.pop()

        while curr_node is not None:
            if self._check_stop_conditions(curr_node):
                curr_node.is_leaf = True
            else:
                curr_node_feature_array = np.array(curr_node.feature_vectors)

                best_model = None
                min_bic = np.inf
                # min_T = np.inf
                n_clusters = -1

                print(f"Current layer: {curr_node.layer}")
                for i in range(min_clusters, max_clusters + 1):
                    # Using diagonals covariance matrices speeds thing up tremendously.
                    gmm = mixture.GaussianMixture(
                        n_components=i, covariance_type="diag").fit(curr_node_feature_array)

                    curr_bic = gmm.bic(curr_node_feature_array)
                    print(f"Trying with {i} clusters, bic: {curr_bic}")
                    if (curr_bic < min_bic):
                        best_model = gmm
                        min_bic = curr_bic
                        n_clusters = i

                print(f"Choosing {n_clusters} clusters")

                node_gmm_parameters = {
                    "covs_array": best_model.covariances_,
                    "means": best_model.means_,
                    "weights": best_model.weights_,
                }

                cluster_asigments = best_model.predict(curr_node_feature_array)

                # Save GMM parameters into curr_node
                curr_node.set_gmm_parameters(node_gmm_parameters)
                vectors_with_clusters = self._asign_vectors_to_clusters(
                    curr_node.ids, curr_node.feature_vectors, cluster_asigments
                )

                sub_nodes = self._create_sub_nodes(
                    vectors_with_clusters,
                    curr_node.ids,
                    curr_node.layer + 1,
                    n_clusters,
                )

                for sub_node in sub_nodes:
                    stack.append(sub_node)

                curr_node.set_sub_nodes(sub_nodes)
                curr_node.n_sub_clusters = n_clusters

            if stack:
                curr_node = stack.pop()
            else:
                return root_node

    def _asign_vectors_to_clusters(self, ids, feature_vectors, cluster_asigments):
        new_data = []
        for i in range(len(ids)):
            new_data_item = (ids[i], feature_vectors[i], cluster_asigments[i])
            new_data.append(new_data_item)

        return new_data

    def _get_cluster_of_data(self, resp_array, index):
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
        print(node.n_feature_vectors)
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

        root_node = Node(
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
        print(d)
        a = np.power(2 * 3.14, d / 2)
        b = np.linalg.det(cov_array) ** -0.5
        c = np.exp(
            -0.5
            * np.matmul(np.matmul((feature_vector - mean).T, np.linalg.inv(cov_array)),
                        (feature_vector - mean),
                        )
        )
        cpd = a * b * c
        return cpd

    # This is basically CPD but with added multiplication with weight
    # (I'm pretty sure, haven't really found any good information on what cpd is.)
    # TODO figure out what multivariate_normal is actully doing with pdf()
    def _calculate_likelihood(self, feature_vector, mean, cov_array, weight):
        likelihood = stats.multivariate_normal(
            mean=mean, cov=cov_array, allow_singular=True).pdf(feature_vector)
        if likelihood == float("inf"):
            likelihood = 1
        return weight * likelihood

    # Function calculates the new mean and cov matrix for a node.
    # These values need to be updated when we want to insert a new,
    # data point. M is the number of data points in the node we are updating
    # TODO break up into 2 functions for clarity.
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
    def _find_leaf_node_for_adding(self, id, query_feature_vector, root_node):

        # TODO update parameters of root node.
        curr_node = root_node
        while not curr_node.is_leaf:
            # TODO use other fucntion than likelihood
            max_cpd = -1
            max_node_index = -1
            max_node_mean
            max_node_cov_array
            for i in range(curr_node.n_sub_clusters):
                mean = curr_node.gmm_parameters["means"][i]
                cov_array = curr_node.gmm_parameters["covs_array"][i]
                cpd = self._calculate_cpd(
                    query_feature_vector, mean, cov_array)
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
            # TODO use setters instead of direct asigment.
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

        curr_node.add_id(id)
        curr_node.add_feature_vector(query_feature_vector)
        curr_node.n_feature_vectors += 1

        return curr_node

    # TODO rename to add_to_cd_tree
    def add_to_cs_tree(self, id, query_feature_vector, root_node):
        # Used to determine if leaf need to be split with new
        # data insertion. Could be set by user.
        gama = 0.1

        curr_node = self._find_leaf_node_for_adding(
            id, query_feature_vector, root_node)

        # Split the leaf node into to nodes if parent n features * gama is
        # smaller than then feature of the leaf node.

        n_feature_vectors_parent = curr_node.n_feature_vectors

        # TODO Can be refractored to look a lot better.
        if curr_node.n_feature_vectors > gama * n_feature_vectors_parent:
            curr_node.is_leaf = False
            gmm = GMM()
            model = gmm.gmm_clustering(np.array(curr_node.feature_vectors), 2)
            curr_node.set_gmm_parameters({
                "covs_array": model.covs_array,
                "means": model.means,
                "weights": model.weights,
            })

            # Might turned this into a function since the same thing
            # happens in init_cd_tree().

            vectors_with_clusters = self._asign_vectors_to_clusters(
                curr_node.ids, curr_node.feature_vectors, model.resp_array
            )

            sub_nodes = self._create_sub_nodes(
                vectors_with_clusters, curr_node.ids, curr_node.layer + 1, 2
            )

            curr_node.set_sub_nodes(sub_nodes)
            curr_node.n_sub_clusters = len(sub_nodes)

        return root_node

    # n_clusters created from current cluster.
    def _create_sub_nodes(
        self, feature_vectors_with_clusters, ids, sub_node_layer, n_clusters
    ):
        sub_nodes = []

        # Outer loop loops throught all of the clusters and creates
        # a new node for each one.
        for index in range(n_clusters):
            self.node_id += 1
            sub_node = Node(layer=sub_node_layer, node_id=self.node_id)

            feature_vectors = []
            ids = []

            # Iterates over feature vectors and puts the ones asigned to current
            # cluster into a list which is then added to the new node.
            for item in feature_vectors_with_clusters:
                if item[2] == index:
                    feature_vectors.append(item[1])
                    ids.append(item[0])

            sub_node.set_ids(ids)
            sub_node.set_feature_vectors(feature_vectors)
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

        ranked_similar_images = sorted(
            similar_images, reverse=True, key=lambda x: x[2])
        return ranked_similar_images

    # K: int. Number of similar images to return
    def find_similar_images(self, root_node, query_feature_vector, n_similar_images):
        stack = []
        stack.append(root_node)
        n_data_points = 0
        similar_data_points = []
        while n_data_points < n_similar_images:
            curr_node = stack.pop()
            if not curr_node.is_leaf:
                resps = []
                for i in range(curr_node.n_sub_clusters):
                    print(i)
                    mean = curr_node.gmm_parameters["means"][i]
                    cov_array = curr_node.gmm_parameters["covs_array"][i]
                    weight = curr_node.gmm_parameters["weights"][i]
                    resp = self._calculate_likelihood(
                        query_feature_vector, mean, cov_array, weight)
                    # Save index of each sub node alongside the cpd. this is done
                    # so that sub nodes can be added in decreasing relevance based
                    # on cpd.
                    resps.append([i, resp])

                print(self._test_resps(resps))

                sorted_resps = sorted(resps, key=lambda x: x[1], reverse=True)

                for item in sorted_resps:
                    stack.append(curr_node.sub_nodes[item[0]])
            else:
                for i in range(curr_node.n_feature_vectors):
                    similar_data_points.append(
                        [curr_node.ids[i], curr_node.feature_vectors[i]]
                    )
                n_data_points = len(similar_data_points)

        return self._rank_images(query_feature_vector, similar_data_points)

    def _test_resps(self, resps):
        print(resps)
        sum = 0
        for resp in resps:
            sum += resp[1]

        return sum


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
