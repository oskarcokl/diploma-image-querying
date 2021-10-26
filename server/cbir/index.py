import argparse
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


# This function is intented to be run only when setting up the initial db.
# WARNING! The function will drop cbir_index table if it already exists!
def init_db(dataset_src):
    command = """
        CREATE TABLE cbir_index (
            id SERIAL PRIMARY KEY,
            image_name VARCHAR(255) NOT NULL,
            image_vector DOUBLE PRECISION[4096]
        )
        """

    if table_operations.table_exists("cbir_index"):
        print("Deleting previous table")
        table_operations.drop_table("cbir_index")

    print("Creating cbir_index table")
    db_connector = DbConnector()
    table_operations.create_table(command, db_connector)

    backbone = Backbone()

    #bar = Bar("Extracting features", max=len(os.listdir(dataset_src)))

    for img_name in os.listdir(dataset_src):
        img_path = os.path.join(dataset_src, img_name)
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        features = backbone.get_features(img_array)

        table_operations.insert_tuple(
            (img_name, features.tolist()), db_connector)
    #     bar.next()

    # bar.finish()


def shift_features(features_list, scalar=1000):
    scaled_features_list = []

    for feature_vector in features_list:
        scaled_features_list.append((feature_vector * scalar).tolist())

    return scaled_features_list


def normalize_sk_learn(feature_list):
    feature_array = np.array(feature_list)
    normalized_feature_array = preprocessing.normalize(
        feature_array, norm="l2")
    normalized_feature_list = normalized_feature_array.tolist()

    return normalized_feature_list


def residuals(feature_list):
    normalized_feature_list = []

    for feature_vector in feature_list:
        normalized_feature_vector = feature_vector - \
            int(np.mean(feature_vector))
        normalized_feature_list.append(normalized_feature_vector.tolist())

    return normalized_feature_list


def min_max_normalization(feature_list):
    scaled_feature_list = []

    for feature_vector in feature_list:
        max = feature_vector.max()
        min = feature_vector.min()
        scaled_feature_vector = [float((x - min) / (max - min)
                                 for x in feature_vector)]
        scaled_feature_list.append(scaled_feature_vector)

    return scaled_feature_list


def normalize_features(feature_list):
    # Feature_list is a list of nd_arrays
    normalized_feature_list = []
    for feature_vector in feature_list:
        normalizd_feature_vector = feature_vector / np.sum(feature_vector)

        normalized_feature_list.append(normalizd_feature_vector.tolist())

    return normalized_feature_list


def reduce_features(feature_list, n_components=100):
    feature_array = np.array(feature_list)
    svd = TruncatedSVD(n_components=n_components)
    svd.fit(feature_array)
    result = svd.transform(feature_array)
    return result.tolist()


def init_cd_tree(data, min_clusters, max_clusters, min_node, l_max):
    command = """
        CREATE TABLE reduced_features (
            id SERIAL PRIMARY KEY,
            image_name VARCHAR(255) NOT NULL,
            image_vector DOUBLE PRECISION[4096]
        )
        """

    if table_operations.table_exists("reduced_features"):
        print("Deleting previous table")
        table_operations.drop_table("reduced_features")

    print("Creating reduced_features table")
    db_connector = DbConnector()
    table_operations.create_table(command, db_connector)

    feature_vectors = [item[2] for item in data]
    img_names = [item[1] for item in data]
    # normalized_feature_vectors = normalize_sk_learn(feature_vectors)
    reduced_feature_vectors = reduce_features(feature_vectors, 200)

    table_operations.insert_tuple_list_reduced(
        list(zip(img_names, reduced_feature_vectors)))

    new_data = []
    for i, item in enumerate(data):
        # Appending tuples here.
        new_data.append((item[0], item[1], reduced_feature_vectors[i]))

    zodb_connector = ZODBConnector()
    zodb_connector.connect()
    cd_tree.init_cd_tree(
        new_data, min_clusters, max_clusters, min_node=min_node, l_max=l_max, zodb_connector=zodb_connector)


def get_data():
    connector = DbConnector()
    connector.cursor.execute("SELECT * FROM cbir_index")
    data = connector.cursor.fetchmany(98000)
    print("Number of indexed images: ", len(data))
    data_array = np.array(data, dtype=object)

    return data_array


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
        data = get_data()
        root_node = init_cd_tree(data, 1, 3, 20, 3)
        print("CD-tree created")
