import argparse
import logging
import os
import sys

import numpy as np
from tensorflow.keras.applications.resnet import preprocess_input
from tensorflow.keras.preprocessing import image
import ZODB.FileStorage
from sklearn.decomposition import TruncatedSVD
from sklearn import preprocessing

# Local application imports
sys.path.insert(0, "../")
from backbone import Backbone
from db_utils.db_connector import DbConnector
from models import cd_tree
from db_utils import table_operations
from db_utils.zodb_connector import ZODBConnector

# Type definitions
Vector = "list[float]"
Data = "list[list[int, string, Vector]]"
FeatureList = "list[Vector]"

"""
This script is used to initialize and create various parts of the system.
Mianly creating database tables and initializing the CD-tree.
"""


def init_db(dataset_src: str):
    """
    Function creates cbir_index database (it drops it if it already exists
    so be carefull). It then populates the table with tuple entries of
    (image_name, image_feautres).

    Parameters
    ----------
    dataset_src : str
        Path to dataset you want to use to populate the table
    """
    command = """
        CREATE TABLE cbir_index (
            id SERIAL PRIMARY KEY,
            image_name VARCHAR(255) NOT NULL,
            image_vector DOUBLE PRECISION[4096]
        )
        """

    if table_operations.table_exists("cbir_index"):
        logging.info("Deleting previous table")
        table_operations.drop_table("cbir_index")

    logging.info("Creating cbir_index table")
    db_connector = DbConnector()
    table_operations.create_table(command, db_connector)

    backbone = Backbone()

    for img_name in os.listdir(dataset_src):
        img_path = os.path.join(dataset_src, img_name)
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        features = backbone.get_features(img_array)

        table_operations.insert_tuple(
            (img_name, features.tolist()), db_connector)


def reduce_features(feature_list: FeatureList, n_components: int = 100) -> FeatureList:
    """
    Reduce features of given features list to specified number of dimensions.
    """
    feature_array = np.array(feature_list)
    svd = TruncatedSVD(n_components=n_components)
    svd.fit(feature_array)
    result = svd.transform(feature_array)
    return result.tolist()


def init_cd_tree(data: Data, min_clusters: int, max_clusters: int, min_node: int, l_max: int):
    """
    Init CD-tree with provided paramaters. While initing also 
    add reduced features to  reduced_features table

    Paramaters
    ----------
    data : [[id, image_name, feature_vector]]
        Array of where each line contains information about an image
    min_clusters : int
        Minum number of clusters on each layer of CD-tree 
    max_clusters : int
        Maximum number of allowed clusters on each layer of CD-tree
    min_node : int
        If number of images in node smaller than this then node becomes leaf
    l_max : int
        Max allowed depth of CD-tree 
    """
    command = """
        CREATE TABLE reduced_features (
            id SERIAL PRIMARY KEY,
            image_name VARCHAR(255) NOT NULL,
            image_vector DOUBLE PRECISION[4096]
        )
        """

    if table_operations.table_exists("reduced_features"):
        logging.info("Deleting previous table")
        table_operations.drop_table("reduced_features")

    logging.info("Creating reduced_features table")
    db_connector = DbConnector()
    table_operations.create_table(command, db_connector)

    feature_vectors = [item[2] for item in data]
    img_names = [item[1] for item in data]
    reduced_feature_vectors = reduce_features(feature_vectors, 200)

    table_operations.insert_tuple_list_reduced(
        list(zip(img_names, reduced_feature_vectors)))

    new_data = []
    for i, item in enumerate(data):
        new_data.append((item[0], item[1], reduced_feature_vectors[i]))

    zodb_connector = ZODBConnector()
    zodb_connector.connect()
    cd_tree.init_cd_tree(
        new_data,
        min_clusters,
        max_clusters,
        min_node=min_node,
        l_max=l_max,
        zodb_connector=zodb_connector
    )


if __name__ == "__main__":
    argParser = argparse.ArgumentParser()
    argParser.add_argument(
        "-d",
        "--dataset",
        help="Path to directory that contains the images to be indexed",
    )
    argParser.add_argument(
        "-IDB",
        "--init-db",
        help="Set if initializing the db for the first time. Will add all pictures in dataset path. Will drop table that was previously initialized.",
        action="store_true",
    )
    argParser.add_argument(
        "-IT",
        "--init-cd-tree",
        help="Set if initializing the cd tree for the first time. Will create a cd tree and store it with ZODB.",
        action="store_true",
    )
    argParser.add_argument(
        "-q",
        "--query",
        help="Path to query image.",
    )

    args = vars(argParser.parse_args())

    if args.get("init_db"):
        init_db(args.get("dataset"))
    elif args.get("init_cd_tree"):
        data = table_operations.get_data_all()
        root_node = init_cd_tree(data, 1, 3, 20, 3)
        logging.info("CD-tree created")
