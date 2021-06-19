# Code written with help from https://github.com/maelfabien/EM_GMM_HMM/blob/master/gmm.py
from re import A
import numpy as np
from scipy.stats import norm, multivariate_normal


class GMM:
    def __init__(self, component_range_min=1, component_range_max=10):
        self.component_range_min = component_range_min
        self.component_range_max = component_range_max

    def gmm_clustering(self, feature_vector_array):
        # for self.curr_components in range(self.component_range_min, self.component_range_max+1):
        self.curr_components = 8
        n_feature_vectors, n_features_length = feature_vector_array.shape

        # This is a responsibility matrix, the matrix describes,
        # the probability resp[j,k] thak j is part of cluster k.
        self.resp_array = np.zeros((n_feature_vectors, self.curr_components))

        # Construct array of random indexes which will later be used as the random starting means.
        # replace=False means that the indexes will be unique.
        the_chosen = np.random.choice(
            n_feature_vectors, self.curr_components, replace=False
        )
        self.means = feature_vector_array[the_chosen]

        # Create array of weights. Weights vector needs to sum to 1,
        # thats why the components at start are 1 / n_components
        self.weights = np.full(self.curr_components, 1 / self.curr_components)

        # Create an array of covariance matrices.
        # Currently using full covariance matrix, check if diagonal is better
        # like in paper
        shape_cov_matrix = self.curr_components, n_features_length, n_features_length
        # TODO try diagonal cov matrix.
        # Here all cov matrices are the same. Not sure if this is intended.
        self.covs_array = np.full(
            shape_cov_matrix, np.cov(feature_vector_array, rowvar=False)
        )

        log_likelihood = 0
        self.log_likelihood_trace = []
        self.has_converged = False

        ll = []

        # 1000 iterations is arbitrary here. Can be set by user in future.
        for i in range(1000):
            new_log_likelihood = self._e_step(feature_vector_array)
            self._do_mstep(feature_vector_array)

            ll.append(new_log_likelihood)

            # 0.001 is an arbitrary tolerance which can later be set by the user.
            if abs(new_log_likelihood - log_likelihood) <= 0.001:
                self.has_converged = True

            log_likelihood = new_log_likelihood
            self.log_likelihood_trace.append(log_likelihood)

        return self, ll

    # The e step we estimate the probability that a certain feature vector
    # belongs to a specific component/cluster.
    def _e_step(self, feature_vector_array):
        log_likelihood = self._compute_log_likelihood(feature_vector_array)

        # Normalize resp array over all possible clutser assignments
        self.resp = self.resp / self.resp.sum(axis=1, keepdims=1)
        return log_likelihood

    def _compute_log_likelihood(self, feature_vector_array):
        for i in range(self.curr_components):
            weight = self.weights[i]
            likelihood = multivariate_normal(self.means[i], self.covs_array[i]).pdf(
                feature_vector_array
            )
            self.resp[:, i] = weight * likelihood

        # Sum all probabilitires of all datapoinst for each cluster
        # use log so we can sum for all datapoinst instead of multiplying it.
        log_likelihood = np.sum(np.log(np.sum(self.resp, axis=1)))
        return log_likelihood

    # Most of calculations are done with matrices but could be just as
    # easily achieved with loops and vectors.
    def m_step(self, feature_vector_array):

        # Sum of responsibilities assigned to each clutser.
        resp_weights = self.resp.sum(axis=0)

        # Calculating new weights from sum of all responsibilities divided by number of features.
        self.weights = resp_weights / feature_vector_array.shape[0]

        # Multiply probabilities with actual feature vector values
        weighted_sum = np.dot(self.resp.T, feature_vector_array)

        # Divide previous mutliplication with sumed probabilities do get means.
        self.means = weighted_sum / resp_weights.reshape(-1, 1)

        for i in range(self.curr_components):
            diff = (feature_vector_array - self.means[i]).T
            weighted_sum = np.dot(self.resp[:, i] * diff, diff.T)
            self.covs[i] = weighted_sum / resp_weights[i]

        return self


if __name__ == "__main__":
    print("Testing GMM class")

    test_list = []

    for i in range(10):
        test_list.append(np.full(4096, i))

    test_array = np.array(test_list)

    myGmm = GMM()
    myGmm.gmm_clustering(test_array)
