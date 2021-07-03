import argparse
import os
from absl.logging import get_absl_handler
import cv2
import numpy as np
from searcher import Searcher
from tensorflow import keras
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras import backend as K
from sklearn.decomposition import TruncatedSVD

import sys
sys.path.insert(0, "../")

from db_connector import DbConnector


def search(query_img_path=None, query_img_list=None, cli=False, dataset=""):
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

    searcher = Searcher()

    if cli:
        try:
            img = image.load_img(query_img_path, target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)

            img_names = find_similar_imgs(
                img_array=img_array, model=model, searcher=searcher
            )

            img_paths = [os.path.join(dataset, img_name)
                         for img_name in img_names]
            print(img_paths)

            show_results(query_img_path, img_paths)
        except Exception as e:
            print(e)
    else:
        query_img_array = np.array(query_img_list)
        img_array = np.expand_dims(query_img_array, axis=0)
        img_paths = find_similar_imgs(
            img_array=img_array, model=model, searcher=searcher
        )
        return img_paths


def show_results(query_img_path, img_paths):
    query_img = cv2.imread(query_img_path)
    query_resized = cv2.resize(query_img, (720, 480))
    cv2.imshow("Query", query_resized)
    cv2.waitKey(0)

    for img_path in img_paths:
        result_img = cv2.imread(img_path)
        result_resized = cv2.resize(result_img, (720, 480))
        cv2.imshow("Result", result_resized)
        cv2.waitKey(0)


def find_similar_imgs(img_array, model, searcher):
    processed_img_array = preprocess_input(img_array)
    get_fc2_layer_output = K.function(
        [model.layers[0].input], model.layers[22].output)
    features_query = get_fc2_layer_output([processed_img_array])[0]

    feature_query_2d = np.array([features_query])

    print(feature_query_2d.shape)

    reduced_feature_query = reduce_features(feature_query_2d)
    print(reduced_feature_query)

    #img_names = searcher.search(features_query.reshape(1, -1), 10)
    # return img_names


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


def get_data():
    connector = DbConnector()
    connector.cursor.execute("SELECT * FROM cbir_index")
    print("Number of indexed images: ", connector.cursor.rowcount)
    data = connector.cursor.fetchall()
    data_array = np.array(data, dtype=object)

    # rand_indexes = np.random.choice(
    #     1909, 1909, replace=False
    # )
    # print(rand_indexes)
    # rand_data = data_array[rand_indexes]
    # print(f"Lenght of subset of data {len(rand_data)}")
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
