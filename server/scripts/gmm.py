# Code written with help from https://github.com/maelfabien/EM_GMM_HMM/blob/master/gmm.py
from re import A
import numpy as np
from scipy.stats import norm, multivariate_normal


"""
Full covariance Gaussina Mixture Model.

Parameters
==========

component_range_min: int
    Minimum number of clutser you want to have in trained model. Default is 1
    Model will train on range (component_range_min to component_range_max)
     of clusters and choose the optimal number.

component_range_max: int
    Maximum number of clutser you want to have in trained model. Default is 10.
    Model will train on range (component_range_min to component_range_max)
     of clusters and choose the optimal number.
"""


class GMM:
    def __init__(self, component_range_min=1, component_range_max=10):
        self.component_range_min = component_range_min
        self.component_range_max = component_range_max

    def get_optimal_clusters(self, feature_vector_array):
        for i in range(self.component_range_min, self.component_range_max + 1):
            gmm_model, ll = self.gmm_clustering(feature_vector_array, i)
            print(ll)
        pass

    def gmm_clustering(self, feature_vector_array, n_components):
        self.curr_components = n_components
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

        print(self.covs_array[0].shape)

        log_likelihood = 0
        self.log_likelihood_trace = []
        self.has_converged = False


        # 1000 iterations is arbitrary here. Can be set by user in future.
        for i in range(1000):
            new_log_likelihood = self._e_step(feature_vector_array)
            self._m_step(feature_vector_array)

            # 0.001 is an arbitrary tolerance which can later be set by the user.
            if abs(new_log_likelihood - log_likelihood) <= 0.01:
                print("Passed tolerance")
                self.has_converged = True
                break

            log_likelihood = new_log_likelihood
            self.log_likelihood_trace.append(log_likelihood)

        return self

    # The e step we estimate the probability that a certain feature vector
    # belongs to a specific component/cluster.
    def _e_step(self, feature_vector_array):
        log_likelihood = self._compute_log_likelihood(feature_vector_array)

        # Normalize resp array over all possible clutser assignments
        self.resp_array = self.resp_array / self.resp_array.sum(axis=1, keepdims=1)
        return log_likelihood

    def _compute_log_likelihood(self, feature_vector_array):
        for i in range(self.curr_components):
            weight = self.weights[i]

            # Getting singular matrix error here. Might just be problem with current data.
            # print(self.covs_array[i])
            likelihood = multivariate_normal(
                mean=self.means[i], cov=self.covs_array[i], allow_singular=True
            ).pdf(feature_vector_array)
            self.resp_array[:, i] = weight * likelihood

        # Sum all probabilitires of all datapoinst for each cluster
        # use log so we can sum for all datapoinst instead of multiplying it.
        log_likelihood = np.sum(np.log(np.sum(self.resp_array, axis=1)))
        return log_likelihood

    # Most of calculations are done with matrices but could be just as
    # easily achieved with loops and vectors.
    def _m_step(self, feature_vector_array):

        # Sum of responsibilities assigned to each clutser.
        resp_weights = self.resp_array.sum(axis=0)

        # Calculating new weights from sum of all responsibilities divided by number of features.
        self.weights = resp_weights / feature_vector_array.shape[0]

        # Multiply probabilities with actual feature vector values
        weighted_sum = np.dot(self.resp_array.T, feature_vector_array)

        # Divide previous mutliplication with sumed probabilities do get means.
        self.means = weighted_sum / resp_weights.reshape(-1, 1)

        for i in range(self.curr_components):
            diff = (feature_vector_array - self.means[i]).T
            weighted_sum = np.dot(self.resp_array[:, i] * diff, diff.T)
            self.covs_array[i] = weighted_sum / resp_weights[i]

        return self


if __name__ == "__main__":
    print("Testing GMM class")

    test_data = np.random.rand(100, 200)

    myGmm = GMM()
    myGmm.get_optimal_clusters(test_data)
