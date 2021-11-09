import logging
import sys

from codetiming import Timer
import numpy as np

sys.path.insert(0, "../")
sys.path.append("./scripts/")
from models import cd_tree
from models.node import Node
from db_utils.zodb_connector import ZODBConnector


# Type definitions
Vector = "list[float]"
FeatureVectors = "list[Vector]"
ImageNames = "list[str]"


class Searcher:
    """
    Searcher is object used to search for similar images with CD-tree.
    """

    def search(self, query_features: Vector, n_similar_images: int, root_node: Node = None) -> ImageNames:
        """
        Search for similar images with CD-tree.

        Paramaters
        ----------
        query_features : Vector
            Feature vector of query image
        n_similar_images : int
            Number of similar images to find and return
        root_node : Node
            Root node of CD-tree

        Returns
        -------
        img_names : ImageNames
            Image names of images similar to query image
        """
        if root_node is None:
            logging.info("No root_node")
            zodb_connector = ZODBConnector()
            zodb_connector.connect()
            root_node = zodb_connector.get_root_node()

        result_images = cd_tree.find_similar_images(
            root_node=root_node,
            query_feature_vector=query_features,
            n_similar_images=n_similar_images
        )

        img_names = []
        for i in range(n_similar_images):
            img_names.append(result_images[i][1])

        return img_names

    def search_force(
            self,
            query_features: Vector,
            feature_vectors: FeatureVectors,
            img_names: ImageNames,
            n_similar_images: int
    ) -> ImageNames:
        """
        Search for similar images with brute force search.

        In this context brute force search referse to comparing
        query image feautres to provided image features currently
        in the database. Cos similarity is used to compare two 
        feature vectors between eachother.

        Parameters
        ----------
        query_features : Vector
            Feature vector of query image
        feature_vectors : FeatureVectors
            Feature vectors to compare query image to
        img_names : ImageNames
            Image names of images to cmopare query image to
        n_similar_images : int
            Number of similar images to find and return

        Returns
        -------
        result_img_names : ImageNames
            Image names of images similar to query image

        """

        distances = self._compute_distances(
            query_features, feature_vectors, img_names)

        ranked_distances = sorted(distances, reverse=True, key=lambda x: x[0])

        result_img_names = []
        for i in range(n_similar_images):
            result_img_names.append(ranked_distances[i][1])

        return result_img_names

    def _compute_distances(self, query_features: Vector, feature_vectors: FeatureVectors, img_names: ImageNames) -> "list[tuple[int, ImageNames]]":
        """
        Compute distances between query feautres and feature vectors.

        Parameters
        ----------
        query_features : Vector
            Feature vector of query image
        feature_vectors : FeatureVectors
            Feature vectors to compare query image to
        img_names : ImageNames
            Image names of images to cmopare query image to

        Returns
        -------
        results : list[tuple[int, ImageNames]]
            List of tuples (d, img_names) where d is the distance of image to query image
        """
        results = []
        for index, feature_vector in enumerate(feature_vectors):
            d = self._compute_cos_similarity(query_features, feature_vector)
            results.append((d, img_names[index]))

        return results

    def _compute_cos_similarity(self, x: int, y: int) -> int:
        """
        Compute cos similarity between vectors x and y.

        Parameters
        ----------
        x : Vector
        y : Vector

        Returns
        -------
        similarity : int
            Cos similarty of x and y vectors
        """
        dot_product = np.dot(x, y)
        magnitude_q = np.linalg.norm(x)
        magnitude_f = np.linalg.norm(y)
        similarity = dot_product / (magnitude_f * magnitude_q)

        return similarity
