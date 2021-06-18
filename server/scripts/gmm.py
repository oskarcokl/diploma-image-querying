# Code written with help from https://github.com/maelfabien/EM_GMM_HMM/blob/master/gmm.py
from os import WEXITED
import numpy as np
from numpy.core.defchararray import replace


class GMM:
    def __init__(self, component_range_min=1, component_range_max=10):
        self.component_range_min = component_range_min
        self.component_range_max = component_range_max

    def gmm_clustering(self, feature_vector_array):
        # for self.curr_components in range(self.component_range_min, self.component_range_max+1):
        self.curr_components = 8
        n_feature_vectors, n_features_length = feature_vector_array.shape

        # This is a responsibility vector, not sure what it does yet.
        self.resp_vector = np.zeros((n_feature_vectors, self.curr_components))

        # Construct array of random indexes which will later be used as the random starting means.
        # replace=False means that the indexes will be unique.
        the_chosen = np.random.choice(
            n_feature_vectors, self.curr_components, replace=False
        )
        self.means = feature_vector_array[the_chosen]

        # Create array of weights. As far as i know now the initial
        # initialization number may be arbitrary.
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
            new_log_likelihood = self._do_estep(feature_vector_array)
            self._do_mstep(feature_vector_array)

            ll.append(new_log_likelihood)

            # 0.001 is an arbitrary tolerance which can later be set by the user.
            if abs(new_log_likelihood - log_likelihood) <= 0.001:
                self.has_converged = True

            log_likelihood = new_log_likelihood
            self.log_likelihood_trace.append(log_likelihood)

        return self, ll



if __name__ == "__main__":
    print("Testing GMM class")

    test_list = []

    for i in range(10):
        test_list.append(np.full(4096, i))

    test_array = np.array(test_list)

    myGmm = GMM()
    myGmm.gmm_clustering(test_array)
