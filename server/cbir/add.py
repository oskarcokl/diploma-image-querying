import argparse
from functools import reduce
import os


from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing import image
from sklearn import preprocessing
from sklearn.decomposition import TruncatedSVD
import numpy as np
import cv2

import sys
sys.path.insert(0, "../")
sys.path.insert(0, "./")
from adder import Adder
from db_utils.zodb_connector import ZODBConnector
from backbone import Backbone
from db_utils.db_connector import DbConnector


def add_cli(img_list):
    backbone = Backbone()

    img_name_list = []
    feature_list = []

    for img_path in img_list:
        img_name = img_path.split("/")[-1]
        save_img_to_disk(img_path, img_name, "./test")
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        features = backbone.get_features(img_array)

        img_name_list.append(img_name)
        feature_list.append(features)

    adder = Adder()

    tuple_list = list(zip(img_name_list, feature_list))

    # ids = []
    # for i in range(len(img_name_list)):
    #     id = adder.add_img_to_db(normalized_features[i], img_name_list[i])
    #     ids.append(id)

    ids = adder.add_img_to_db(tuple_list)

    zodb_connector = ZODBConnector()
    zodb_connector.connect("./cd_tree.fs")

    add_to_cd_tree(ids, np.array(feature_list),
                   img_name_list, adder, zodb_connector)

    zodb_connector.disconnect()


def add(decoded_images, root_node=None):
    adder = Adder()

    feature_list = []
    image_names = []

    for decoded_image in decoded_images:
        feature_list.append(decoded_image[1])
        image_names.append(decoded_image[0])

    tuple_list = list(zip(image_names, feature_list))

    ids = adder.add_img_to_db(tuple_list)

    add_to_cd_tree(ids, np.array(feature_list),
                   image_names, adder, root_node=root_node)


def add_to_cd_tree(ids, feature_vectors, img_name_list, adder, root_node=None):
    if root_node is None:
        zodb_connector = ZODBConnector()
        zodb_connector.connect()
        root_node = zodb_connector.get_root_node()

    for i in range(len(feature_vectors)):
        root_node = adder.add_to_cd_tree(
            ids[i], feature_vectors[i], img_name_list[i], root_node)
        zodb_connector.save_cd_tree(root_node)


def reduce_features(add_features, n_components=100):
    feature_vectors = get_data()
    feature_vectors_plus = np.append(
        feature_vectors, np.array(add_features), axis=0)

    feature_array = np.array(feature_vectors_plus)
    svd = TruncatedSVD(n_components=n_components)
    svd.fit(feature_array)
    result = svd.transform(feature_array)
    add_reduced = result[len(result) - len(add_features):]
    return add_reduced


def save_img_to_disk(img_path, img_name, path):
    img = cv2.imread(img_path)
    save_path = os.path.join(path, img_name)
    print(save_path)
    result = cv2.imwrite(save_path, img)
    if result:
        print("Saved image successfully. :^)")
    else:
        print("Error while saving image.")


def get_data():
    connector = DbConnector()
    connector.cursor.execute("SELECT * FROM cbir_index")
    print("Number of indexed images: ", connector.cursor.rowcount)
    data = connector.cursor.fetchall()
    data_array = np.array(data, dtype=object)

    feature_vectors = data_array[:, 2]
    result = [np.array(feature_vector) for feature_vector in feature_vectors]
    return np.array(result)


def normalize_sk_learn(feature_list):
    feature_array = np.array(feature_list)
    normalized_feature_array = preprocessing.normalize(
        feature_array, norm="max")
    normalized_feature_list = normalized_feature_array.tolist()

    return normalized_feature_list


if __name__ == "__main__":

    argParser = argparse.ArgumentParser()
    argParser.add_argument("-i", "--images", nargs="+", required=True,
                           help="Pass paths to images you want to add to index.")

    args = vars(argParser.parse_args())

    add_cli(args["images"])
