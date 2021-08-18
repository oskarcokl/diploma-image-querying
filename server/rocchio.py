import numpy as np


def make_new_query(query_features, related_images, non_related_images, a=1, b=0.8, c=0.1):

    related_sum = np.zeros(query_features.size)
    for related_features in related_images:
        related_sum = related_sum + related_features

    non_related_sum = np.zeros(query_features.size)
    for non_related_features in non_related_images:
        non_related_sum = non_related_sum + non_related_features

    new_query = (a * query_features) + (b * 1 / len(related_images) *
                                        related_sum) - (c * 1 / len(non_related_images) * non_related_sum)

    return new_query
