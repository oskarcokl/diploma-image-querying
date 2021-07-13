import argparse
import logging
import os

import cv2
import numpy as np
from searcher import Searcher
from tensorflow import keras
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras import backend as K
from sklearn.decomposition import TruncatedSVD
from sklearn import preprocessing
from codetiming import Timer

import sys
sys.path.insert(0, "../")
sys.path.insert(0, "./")

from db_utils.db_connector import DbConnector
from term_colors import TerminalColors

T_FEAT_REDUCTION = 0
T_SEARCH = 0
T_NORMALIZATION = 0
T_DB = 0
T_MODEL = 0
T_ALL = 0


def search(query_img_path=None, query_img_list=None, cli=False, dataset=""):
    t_all = Timer(name="All")
    t_all.start()
    t_model = Timer(name="Model")
    t_model.start()

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

    T_MODEL = t_model.stop()
    searcher = Searcher()

    if cli:
        try:
            img = image.load_img(query_img_path, target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)

            img_names = find_similar_imgs(
                img_array=img_array, model=model, searcher=searcher
            )

            # img_paths = [os.path.join(dataset, img_name)
            #  for img_name in img_names]

            # show_results(query_img_path, img_paths)
            T_ALL = t_all.stop()
        except Exception as e:
            print(e)
    else:
        query_img_array = np.array(query_img_list)
        img_array = np.expand_dims(query_img_array, axis=0)

        img_names = find_similar_imgs(
            img_array=img_array, model=model, searcher=searcher
        )
        return img_names


def show_results(query_img_path, img_paths):
    query_img = cv2.imread(query_img_path)
    query_resized = cv2.resize(query_img, (720, 480))
    cv2.imshow("Query", query_resized)

    for img_path in img_paths:
        result_img = cv2.imread(img_path)
        result_resized = cv2.resize(result_img, (720, 480))
        cv2.imshow("Result", result_resized)
        cv2.waitKey(0)


def find_similar_imgs(img_array, model, searcher):
    processed_img_array = preprocess_input(img_array)
    get_fc2_layer_output = K.function(
        [model.layers[0].input], model.layers[21].output)
    features_query = get_fc2_layer_output([processed_img_array])[0]

    t = Timer(name="Normalization", logger=None)
    t.start()
    normalized_feature_query = preprocessing.normalize(
        features_query.reshape(1, -1), norm="max")
    print("hello")
    norm_time = t.stop()

    t = Timer(name="Feature reduction", logger=None)
    t.start()
    reduced_feature_query = reduce_features(normalized_feature_query, 40)
    feat_reduction_time = t.stop()

    img_names, search_time = searcher.search(reduced_feature_query, 10)
    return img_names, search_time, norm_time


@Timer(name="Feature reduction", text="Reducing features took {:.4f}s")
def reduce_features(query_features, n_components=100):
    feature_vectors = get_data()
    feature_vectors_plus = np.append(
        feature_vectors, np.array(query_features), axis=0)

    feature_array = np.array(feature_vectors_plus)
    svd = TruncatedSVD(n_components=n_components)
    svd.fit(feature_array)
    result = svd.transform(feature_array)
    query_reduced = result[len(result) - 1]
    return query_reduced


@Timer(name="Database", text="Loading from database took {:.4f}s")
def get_data():
    connector = DbConnector()
    connector.cursor.execute("SELECT * FROM cbir_index")
    data = connector.cursor.fetchall()
    data_array = np.array(data, dtype=object)

    feature_vectors = data_array[:, 2]
    result = [np.array(feature_vector) for feature_vector in feature_vectors]
    return np.array(result)


if __name__ == "__main__":
    argParser = argparse.ArgumentParser()
    argParser.add_argument(
        "-q", "--query", required=True, help="Path to the query image"
    )
    argParser.add_argument(
        "-T",
        "--terminal",
        help="Use if you want to call the script from a CLI.",
        action="store_true",
    )
    argParser.add_argument(
        "-d",
        "--dataset",
        help="Path to where images are being stored"
    )
    args = vars(argParser.parse_args())

    print(args["terminal"])

    search(query_img_path=args["query"],
           cli=args["terminal"], dataset=args["dataset"])
