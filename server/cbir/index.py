import argparse
import os
import sys
import pickle

import psycopg2
import numpy as np
from tensorflow import keras
from tensorflow.keras.applications.resnet import preprocess_input
from tensorflow.keras.preprocessing import image
import ZODB
import ZODB.FileStorage
import BTrees
import transaction
from sklearn.decomposition import TruncatedSVD
from sklearn import preprocessing
from progress.bar import Bar

# Local application imports
sys.path.insert(0, "../")
from backbone import Backbone
from db_utils.db_connector import DbConnector
from models import cd_tree
from db_utils import table_operations


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
    table_operations.create_table(command)

    backbone = Backbone()

    img_name_list = []
    feature_list = []

    #bar = Bar("Extracting features", max=len(os.listdir(dataset_src)))

    for img_name in os.listdir(dataset_src):
        img_path = os.path.join(dataset_src, img_name)
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        features = backbone.get_features(img_array)

        img_name_list.append(img_name)
        feature_list.append(features.tolist())
    #     bar.next()

    # bar.finish()

    tuple_list = list(zip(img_name_list, feature_list))

    table_operations.insert_tuple_list(tuple_list)


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
    feature_vectors = [item[2] for item in data]
    # normalized_feature_vectors = normalize_sk_learn(feature_vectors)
    reduced_feature_vectors = reduce_features(feature_vectors, 140)
    new_data = []
    for i, item in enumerate(data):
        # Appending tuples here.
        new_data.append((item[0], item[1], reduced_feature_vectors[i]))

    # TODO Change back to new data
    root_node = cd_tree.init_cd_tree(
        new_data, min_clusters, max_clusters, min_node=min_node, l_max=l_max)
    return root_node


def save_cd_tree(root_node):
    storage = ZODB.FileStorage.FileStorage(
        "cd_tree.fs", blob_dir="cd_tree_blob")
    db = ZODB.DB(storage)
    connection = db.open()
    root = connection.root
    root.cd_tree = BTrees.OOBTree.BTree()
    root.cd_tree["root_node"] = root_node
    transaction.commit()


def get_cd_tree_from_storage():
    storage = ZODB.FileStorage.FileStorage("cd_tree.fs")
    db = ZODB.DB(storage)
    connection = db.open()
    root = connection.root

    print(len(root.cd_tree["root_node"].sub_nodes[0].ids))
    print(root.cd_tree["root_node"].sub_nodes[0].n_feature_vectors)


def make_test_query_feature(query_img_path):
    model = load_model()

    img = image.load_img(query_img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    processed_img_array = preprocess_input(img_array)
    get_fc2_layer_output = K.function(
        [model.layers[0].input], model.layers[22].output)
    features_query = get_fc2_layer_output([processed_img_array])[0]

    with open("query_features", "wb") as f:
        pickle.dump(features_query, f)
    return ":)"


def load_model():
    model = None
    if os.path.isdir("./vgg16"):
        print("Model already downloaded loading from disk.")
        model = keras.models.load_model("./vgg16")
    else:
        print("Downloading model.")
        model = VGG16(
            weights="imagenet",
        )
        print("Saving model to disk.")
        model.save("./vgg16")

    return model


def get_data():
    connector = DbConnector()
    connector.cursor.execute("SELECT * FROM cbir_index")
    data = connector.cursor.fetchmany(95000)
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
        "-IQ",
        "--init-query",
        help="Create a query and save it to a file to load it later.",
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
        root_node = init_cd_tree(data, 1, 14, 150, 14)
        save_cd_tree(root_node)
    elif args.get("init_query"):
        make_test_query_feature(args.get("query"))
    else:
        get_cd_tree_from_storage()
