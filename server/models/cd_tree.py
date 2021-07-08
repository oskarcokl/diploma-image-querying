import numpy as np
from numpy.core.numeric import Inf
from sklearn import mixture

from .gmm import GMM
from .node import Node


def init_cd_tree(
    data,
    min_clusters=1,
    max_clusters=10,
    min_node=20,
    l_max=5,
    # tolerance=0.001,
    # n_iters=1000,
):
    """
    Function that initializes the CDTree and returns it.

    Parameters
    ----------

    data: [id: int, img_src: string, feature_vector: [int]]
    """

    node_id = 0
    stack = []
    root_node = _generate_root_node(data, node_id)
    stack.append(root_node)
    curr_node = stack.pop()

    while curr_node is not None:
        if _check_stop_conditions(curr_node, min_node, l_max):
            leaf_feature_vectors = _get_feature_vectors_by_id(
                data, curr_node.ids)
            leaf_img_names = _get_img_names_by_id(
                data, curr_node.ids)

            curr_node.make_leaf(leaf_feature_vectors, leaf_img_names)
        else:
            print(curr_node.ids)

            curr_node_feature_array = np.array(
                _get_feature_vectors_by_id(data, curr_node.ids))

            print(curr_node_feature_array.shape)

            best_model = None
            min_bic = np.inf
            n_clusters = -1

            #print(f"Current layer: {curr_node.layer}")
            for i in range(min_clusters, max_clusters + 1):
                # Using diagonals covariance matrices speeds thing up tremendously.
                gmm = mixture.GaussianMixture(
                    n_components=i, covariance_type="diag").fit(curr_node_feature_array)

                curr_bic = gmm.bic(curr_node_feature_array)
                #print(f"Trying with {i} clusters, bic: {curr_bic}")
                if (curr_bic < min_bic):
                    best_model = gmm
                    min_bic = curr_bic
                    n_clusters = i

            #print(f"Choosing {n_clusters} clusters")

            node_gmm_parameters = {
                "covs_array": best_model.covariances_,
                "means": best_model.means_,
                "weights": best_model.weights_,
            }

            cluster_asigments = _predict(
                curr_node_feature_array, best_model.means_, best_model.covariances_)

            # Save GMM parameters into curr_node
            curr_node.set_gmm_parameters(node_gmm_parameters)

            ids_with_clusters = _asign_ids_to_clusters(
                curr_node.ids, cluster_asigments)

            sub_nodes = _create_sub_nodes(
                ids_with_clusters,
                curr_node.layer + 1,
                n_clusters,
                node_id
            )

            for sub_node in sub_nodes:
                stack.append(sub_node)

            curr_node.set_sub_nodes(sub_nodes)
            curr_node.n_sub_clusters = n_clusters

        if stack:
            curr_node = stack.pop()
        else:
            return root_node


def _asign_ids_to_clusters(ids, cluster_asigments):

    new_data = []
    for i in range(len(ids)):
        new_data_item = (ids[i], cluster_asigments[i])
        new_data.append(new_data_item)

    return new_data


def _predict(feature_vectors, means, covs_array):
    # Func returns asigmnets of feature vectors to clusters.
    assigments = []

    for feature_vector in feature_vectors:
        cpds = _compute_cpds(feature_vector, means, covs_array)
        max_cpd_cluster = _get_max_cpd_index(cpds)
        assigments.append(max_cpd_cluster)

    return assigments


def _compute_cpds(feature_vector, means, covs_array):
    cpds = []
    for i in range(len(means)):
        cpd = _compute_cpd(
            feature_vector, means[i], covs_array[i])
        cpds.append(cpd)

    return cpds


def _get_max_cpd_index(cpds):
    return cpds.index(max(cpds))


def _get_cluster_of_data(resp_array, index):
    n_clusters = resp_array.shape[1]
    # Probability (responsibility) can never be smaller than 0.
    max_resp = -1
    cluster = -1

    for i in range(n_clusters):
        if resp_array[index][i] > max_resp:
            max_resp = resp_array[index][i]
            cluster = i

    return cluster


def _get_feature_vectors_and_imgs_by_id(data, ids):
    features = []
    img_names = []
    data_start_index = 0
    for id in enumerate(ids):
        for i in range(data_start_index, len(data)):
            if data[i][0] == id:
                features.append(data[i][2])
                img_names.append(data[i][1])
                data_start_index = i + 1
                break
    return (features, img_names)


def _get_img_names_by_id(data, ids):
    names = []
    for i, id in enumerate(ids):
        if data[i][0] == id:
            names.append(data[i][1])
    return names


def _check_stop_conditions(node, min_node, l_max):
    # If stop conditions have been met the functino returns true.
    # Else it retusn false.
    if node.gmm_parameters and len(node.gmm_parameters["weights"]) == 1:
        return True
    elif node.n_feature_vectors < min_node:
        return True
    elif node.layer >= l_max:
        return True

    return False


def _generate_root_node(data, node_id):
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
        node_id=node_id,
    )
    return root_node


def _compute_cpd(feature_vector, mean, cov_array):
    # (2π)^(−d/2) * |Σi|^(−1/2) * exp(−1/2(X−μi)TΣ−1i(X−μi)).
    # Function calculates the above equation
    # All the bad var names are made on a saturday ad 20:55 pls forgive me.
    # TODO first component is probably not useful. Calculate determinante in
    # a better way since it's diagonal

    cov_array_dig = np.diag(cov_array)
    d = len(feature_vector)
    half_d = d / 2
    two_pi = 2 * np.pi
    a = np.power(two_pi, half_d)
    b = np.linalg.det(cov_array_dig)
    b_power = b ** -0.5
    if np.isnan(b_power):
        b_power = 0

    c = np.exp(
        -0.5
        * np.matmul(np.matmul((feature_vector - mean).T, np.linalg.inv(cov_array_dig)),
                    (feature_vector - mean),
                    )
    )
    first_part = a * b_power
    cpd = first_part * c
    return cpd


def _compute_means_and_covs(m, feature_vector, mean, cov_array):
    # Function calculates the new mean and cov matrix for a node.
    # These values need to be updated when we want to insert a new,
    # data point. M is the number of data points in the node we are updating

    new_mean = _compute_mean(m, mean, feature_vector)
    new_cov_array = _compute_cov(
        m, feature_vector, mean, cov_array)
    return new_mean, new_cov_array


def _compute_mean(m, mean, feature_vector):
    new_mean = ((m / (m + 1)) * mean) + \
        ((1 / (m + 1)) * feature_vector)
    return new_mean


def _compute_cov(m, feature_vector, mean, cov_array):
    cov_mat = np.diag(cov_array)
    new_cov_array = (((m - 1) / m) * cov_mat) + (
        (1 / (m + 1))
        * np.matmul((feature_vector - mean), (feature_vector - mean).T)
    )

    diag_values = np.diag(new_cov_array)
    return diag_values


def _compute_covs(m, feature_vector, means, covs_array):
    n_clusters = len(means)
    n_features = len(feature_vector)

    covs = np.zeros((n_clusters, n_features))

    for i in range(n_clusters):
        covs[i] = _compute_cov(m, feature_vector, means[i], covs_array[i])

    return covs


def _compute_means(m, means, feature_vector):
    n_clusters = len(means)
    n_features = len(feature_vector)

    new_means = np.zeros((n_clusters, n_features))

    for i in range(n_clusters):
        new_means[i] = _compute_mean(m, means[i], feature_vector)

    return new_means


def _find_leaf_node_for_adding(id, feature_vector, img_name, root_node):
    # Algorithm finds leaf node of query_feature_vector. It finds
    # the leaf by calculating cpd's for each subnode and choosing
    # the subnode with the highest cpd.

    # Means and covs of root do not get update. This might hurt
    # results in the long run. Maybe tree need to be rebuilt once
    # in a while. The algorthim doesn't call for it but I might still
    # do it.
    curr_node = root_node
    n_feature_vectors_parent = 0
    while not curr_node.is_leaf:
        n_feature_vectors_parent = curr_node.n_feature_vectors

        means = curr_node.gmm_parameters["means"]
        cov_arrays = curr_node.gmm_parameters["covs_array"]

        cpds = _compute_cpds(feature_vector, means, cov_arrays)

        max_cpd_index = _get_max_cpd_index(cpds)

        sub_node = curr_node.sub_nodes[max_cpd_index]
        sub_node.ids.append(id)
        sub_node.n_feature_vectors += 1

        if not sub_node.is_leaf:
            _update_gmm_params(sub_node, feature_vector)

        curr_node = sub_node

    curr_node.add_feature_vector(feature_vector.tolist())
    curr_node.add_img(img_name)

    return (curr_node, n_feature_vectors_parent)


def _update_gmm_params(sub_node, query_feature_vector):
    n_feature_vectors = sub_node.n_feature_vectors
    old_means = sub_node.gmm_parameters["means"]
    old_covs_array = sub_node.gmm_parameters["covs_array"]

    # Calculate mean and cov and update them for the node
    # with the max cpd.
    new_means = _compute_means(
        n_feature_vectors, old_means, query_feature_vector)
    new_covs_array = _compute_covs(
        n_feature_vectors, query_feature_vector, old_covs_array, old_means)

    sub_node.set_means(new_means)
    sub_node.set_covs_array(new_covs_array)


def add_to_cd_tree(id, feature_vector, img_name, root_node):
    # Used to determine if leaf need to be split with new
    # data insertion. Could be set by user.
    gama = 0.1

    node, n_feature_vectors_parent = _find_leaf_node_for_adding(
        id, feature_vector, img_name, root_node)

    # Split the leaf node into two nodes if parent n features * gama is
    # smaller than then feature of the leaf node.
    if node.n_feature_vectors > gama * n_feature_vectors_parent:
        feature_vectors_array = np.array(node.feature_vectors)

        gmm = mixture.GaussianMixture(
            n_components=2, covariance_type="diag").fit(feature_vectors_array)

        node.set_gmm_parameters({
            "covs_array": gmm.covariances_,
            "means": gmm.means_,
            "weights": gmm.weights_,
        })

        print(f"New parent node: {node}")

        cluster_asigments = _predict(
            feature_vectors_array, gmm.means_, gmm.covariances_)

        ids_with_clusters = _asign_ids_to_clusters(
            node.ids, cluster_asigments)

        # Node ID could be removed, or I have to somehow presistently store the curr
        # counter.
        sub_nodes = _create_sub_nodes(
            ids_with_clusters,
            node.layer + 1,
            2,
            0
        )

        data = []
        for i in range(node.ids):
            data.append((node.ids[i], node.img_names[i],
                        node.feature_vectors[i]))

        for sub_node in sub_nodes:
            leaf_feature_vectors = _get_feature_vectors_by_id(
                data, sub_node.ids)
            leaf_img_names = _get_img_names_by_id(
                data, sub_node.ids)
            sub_node.make_leaf(leaf_feature_vectors, leaf_img_names)

        print(f"Sub node 1: {sub_nodes[0]}")
        print(f"Sub node 2: {sub_nodes[1]}")

        node.set_sub_nodes(sub_nodes)
        node.n_sub_clusters = 2
        node.make_inner_node()

    return root_node


def _create_sub_nodes(
    ids_with_clusters, sub_node_layer, n_clusters, node_id
):
    sub_nodes = []

    # Outer loop loops throught all of the clusters and creates
    # a new node for each one.
    for index in range(n_clusters):
        node_id += 1
        sub_node = Node(layer=sub_node_layer, node_id=node_id)

        ids = []

        # Iterates over feature vectors and puts the ones asigned to current
        # cluster into a list which is then added to the new node.
        for item in ids_with_clusters:
            if item[1] == index:
                ids.append(item[0])

        sub_node.set_ids(ids)
        sub_node.n_feature_vectors = len(ids)
        sub_nodes.append(sub_node)

    return sub_nodes


def _compute_eucledian_distance(query_feature_vector, feature_vector):
    result_array = np.linalg.norm(
        np.array(query_feature_vector) - np.array(feature_vector)
    )
    return result_array.tolist()


def _compute_cosine_similarity(query_feature_vector, feature_vector):
    dot_product = np.dot(query_feature_vector, feature_vector)
    magnitude_q = np.linalg.norm(query_feature_vector)
    magnitude_f = np.linalg.norm(feature_vector)
    result = dot_product / (magnitude_f * magnitude_q)
    return result


def _rank_images(query_feature_vector, similar_images):
    for i in range(len(similar_images)):
        d = _compute_cosine_similarity(
            query_feature_vector, similar_images[i][1]
        )
        similar_images[i].append(d)

    ranked_similar_images = sorted(
        similar_images, reverse=False, key=lambda x: x[3])

    return ranked_similar_images

# K: int. Number of similar images to return


def find_similar_images(root_node, query_feature_vector, n_similar_images):
    stack = []
    stack.append(root_node)
    n_data_points = 0
    similar_data_points = []
    while n_data_points < n_similar_images:
        curr_node = stack.pop()
        if not curr_node.is_leaf:
            means = curr_node.gmm_parameters["means"]
            cov_array = curr_node.gmm_parameters["covs_array"]

            cvds = _compute_cpds(
                query_feature_vector, means, cov_array)

            cvds_index = [(i, cvd) for i, cvd in enumerate(cvds)]

            sorted_cvds = sorted(
                cvds_index, key=lambda x: x[1], reverse=False)

            print(sorted_cvds)

            for item in sorted_cvds:
                stack.append(curr_node.sub_nodes[item[0]])
        else:
            for i in range(curr_node.n_feature_vectors):
                similar_data_points.append(
                    [curr_node.ids[i],
                        curr_node.feature_vectors[i],
                        curr_node.img_names[i]]
                )
            n_data_points = len(similar_data_points)

    ranked_images = _rank_images(
        query_feature_vector, similar_data_points)
    return ranked_images


# Used for quick testing during developement.
if __name__ == "__main__":
    print("Testing CDTree class")

    # Test data
    data_list = []

    for id in range(1, 21):
        rand_feature_vector = np.random.rand(40)
        temp_list = [id, "img_src", rand_feature_vector]
        data_list.append(temp_list)

    root = init_cd_tree(data_list)

    print(root)
    print(root.sub_nodes[0])
    print(root.sub_nodes[1])
