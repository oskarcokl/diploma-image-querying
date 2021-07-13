import argparse
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
from csv_writer import save_to_csv
from term_colors import TerminalColors

T_FEAT_REDUCTION = 0
T_SEARCH = 0
T_NORMALIZATION = 0
T_DB = 0
T_MODEL = 0
T_ALL = 0


def search(query_img_path=None, query_img_list=None, cli=False, dataset=""):
    t_all = Timer(name="All", logger=None)
    t_all.start()
    t_model = Timer(name="Model", logger=None)
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

    global T_MODEL
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
            global T_ALL
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

    t_norm = Timer(name="Normalization", logger=None)
    t_norm.start()
    normalized_feature_query = preprocessing.normalize(
        features_query.reshape(1, -1), norm="max")
    global T_NORMALIZATION
    T_NORMALIZATION = t_norm.stop()

    reduced_feature_query = reduce_features(normalized_feature_query, 40)

    global T_SEARCH
    img_names, T_SEARCH = searcher.search(reduced_feature_query, 10)
    return img_names


def reduce_features(query_features, n_components=100):
    t_db = Timer(name="Database", logger=None)
    t_db.start()

    feature_vectors = get_data()

    global T_DB
    T_DB = t_db.stop()

    t_feat_reduce = Timer(name="Feature reduction", logger=None)
    t_feat_reduce.start()

    feature_vectors_plus = np.append(
        feature_vectors, np.array(query_features), axis=0)
    feature_array = np.array(feature_vectors_plus)
    svd = TruncatedSVD(n_components=n_components)
    svd.fit(feature_array)
    result = svd.transform(feature_array)
    query_reduced = result[len(result) - 1]

    global T_FEAT_REDUCTION
    T_FEAT_REDUCTION = t_feat_reduce.stop()
    return query_reduced


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

    row = [T_MODEL, T_NORMALIZATION, T_DB, T_FEAT_REDUCTION, T_SEARCH, T_ALL]
    print(row)
    save_to_csv("../experiments/oxford.csv", row)


print(T_MODEL, T_NORMALIZATION, T_DB, T_FEAT_REDUCTION, T_SEARCH, T_ALL)
