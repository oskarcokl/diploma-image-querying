import argparse
import os

import cv2
import numpy as np
from tensorflow.keras.applications.resnet import preprocess_input
from tensorflow.keras.preprocessing import image
from sklearn.decomposition import TruncatedSVD
from sklearn import preprocessing
from codetiming import Timer

import sys
sys.path.insert(0, "../")
sys.path.insert(0, "./")
sys.path.insert(0, "./cbir/")

from db_utils.db_connector import DbConnector
from csv_writer import save_to_csv
from backbone import Backbone
from searcher import Searcher

T_SEARCH = 0
T_MODEL = 0
T_ALL = 0
T_FEAT_REDUCTION = 0
T_DB = 0


def search(
    query_img_path=None,
    query_img_list=None,
    cli=False,
    backbone=None,
    searcher=None,
    query_features=None,
    n_images=10,
    dataset=None,
    feature_vectors=None,
    root_node=None
):

    t_all = Timer(name="All", logger=None)
    t_all.start()
    t_model = Timer(name="Model", logger=None)
    t_model.start()

    if not backbone and not query_features:
        print("No backbone")
        backbone = Backbone()

    if not searcher:
        searcher = Searcher()

    global T_MODEL
    T_MODEL = t_model.stop()

    if cli:
        img = image.load_img(query_img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)

        print("Finding similar images")

        img_names = find_similar_imgs(
            img_array=img_array,
            backbone=backbone,
            searcher=searcher,
            n_images=n_images,
            feature_vectors=np.array(feature_vectors),
            root_node=root_node
        )

        # TODO uncomment
        # img_paths = [os.path.join(dataset, img_name)
        #              for img_name in img_names]

        # show_results(query_img_path, img_paths)
        # global T_ALL
        # T_ALL = t_all.stop()

        # TODO Only to be used while evaluating system.
        ranked_img_names = []
        for i, img_name in enumerate(img_names):
            ranked_img_name = " ".join((str(i), img_name))
            ranked_img_names.append(ranked_img_name)

        line = " ".join(ranked_img_names)
        global T_ALL
        T_ALL = t_all.stop()
        write_time()
        return line

    else:
        if query_features:
            img_names = find_similar_imgs(
                backbone=backbone,
                searcher=searcher,
                features_query=query_features,
                n_images=n_images,
                feature_vectors=feature_vectors,
                root_node=root_node
            )
            return img_names
        else:
            query_img_array = np.array(query_img_list)
            img_array = np.expand_dims(query_img_array, axis=0)

            img_names = find_similar_imgs(
                img_array=img_array,
                backbone=backbone,
                searcher=searcher,
                n_images=n_images,
                feature_vectors=feature_vectors,
                root_node=root_node
            )
            return img_names


def brute_force_search(query_img_path=None, backbone=None, searcher=None, n_images=10, feature_vectors=np.array([]), dataset="", img_names=None, reduced_feature_vectors=None):
    t_all = Timer(name="All", logger=None)
    t_all.start()

    t_model = Timer(name="Model", logger=None)
    t_model.start()
    if not backbone:
        backbone = Backbone()
    global T_MODEL
    T_MODEL = t_model.stop()

    if not searcher:
        searcher = Searcher()

    try:
        img = image.load_img(query_img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)

        result_img_names = find_similar_imgs_force(
            img_array=img_array,
            backbone=backbone,
            searcher=searcher,
            feature_vectors=feature_vectors,
            n_images=n_images,
            img_names=img_names,
            reduced_feature_vectors=reduced_feature_vectors
        )

        # img_paths = [os.path.join(dataset, img_name)
        #              for img_name in img_names]
        # show_results(query_img_path, img_paths)

        ranked_img_names = []
        for i, img_name in enumerate(result_img_names):
            ranked_img_name = " ".join((str(i), img_name))
            ranked_img_names.append(ranked_img_name)

        line = " ".join(ranked_img_names)
        global T_ALL
        T_ALL = t_all.stop()

        write_time()
        return line

    except Exception as e:
        print(e)


def find_similar_imgs_force(img_array, backbone: Backbone, searcher, feature_vectors=None, reduced_feature_vectors=None, n_images=10, img_names=None):
    print("Im forcing")
    processed_img_array = preprocess_input(img_array)

    features_query = backbone.get_features(processed_img_array)

    if feature_vectors.size <= 0:
        print("no feature vectors")
        feature_vectors = get_feature_vectors()

    n_features = 140

    # We are almost doing the same operation twice. but since we are adding
    # the query features in reduce features query. The reduction would not be the same.
    if reduced_feature_vectors.size <= 0:
        print("Reducing feature vectors")
        svd = TruncatedSVD(n_components=n_features)
        svd.fit(feature_vectors)
        reduced_feature_vectors = svd.transform(feature_vectors)

    if len(img_names) <= 0:
        print("Getting img names")
        img_names = get_img_names()

    # reduced_feature_query = reduce_features_query(
    #     features_query.reshape(1, -1), feature_vectors, n_features)

    global T_SEARCH
    result_img_names, T_SEARCH = searcher.search_force(
        features_query.reshape(1, -1), reduced_feature_vectors, img_names, n_images)
    return result_img_names


def find_similar_imgs(
    backbone,
    searcher,
    features_query=None,
    img_array=None,
    n_images=10,
    feature_vectors=None,
    root_node=None
):
    global T_SEARCH
    if not features_query:
        processed_img_array = preprocess_input(img_array)
        features_query_array = backbone.get_features(processed_img_array)
    else:
        features_query_array = np.array(features_query)

    if feature_vectors is None:
        feature_vectors = get_feature_vectors()

    reduced_features_query = reduce_features_query(
        features_query_array.reshape(1, -1), feature_vectors, 140)

    del feature_vectors

    img_names, T_SEARCH = searcher.search(
        reduced_features_query, n_images, root_node=root_node)
    del searcher
    return img_names


def reduce_features_query(query_features, feature_vectors, n_components=100):
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


def get_feature_vectors():
    print("Getting feature vectors")
    t_db_timer = Timer(name="Db timer", logger=None)
    t_db_timer.start()
    connector = DbConnector()
    connector.cursor.execute("SELECT * FROM cbir_index")
    data = connector.cursor.fetchall()
    data_array = np.array(data, dtype=object)

    feature_vectors = data_array[:, 2]
    result = np.array([np.array(feature_vector)
                       for feature_vector in feature_vectors])
    global T_DB
    T_DB = t_db_timer.stop()
    return result


def get_img_names():
    connector = DbConnector()
    connector.cursor.execute("SELECT * FROM cbir_index")
    data = connector.cursor.fetchall()
    data_array = np.array(data, dtype=object)

    img_names = data_array[:, 1]
    result = [np.array(img_name) for img_name in img_names]
    return np.array(result)


def get_names_and_features():
    connector = DbConnector()
    connector.cursor.execute("SELECT * FROM cbir_index")
    data = connector.cursor.fetchall()
    data_array = np.array(data, dtype=object)

    img_names = data_array[:, 1]
    result_img_names = [np.array(img_name) for img_name in img_names]

    feature_vectors = data_array[:, 2]
    result_feature_vectors = [np.array(feature_vector)
                              for feature_vector in feature_vectors]
    return np.array(result_img_names), np.array(result_feature_vectors)


def show_results(query_img_path, img_paths):
    query_img = cv2.imread(query_img_path)
    query_resized = cv2.resize(query_img, (720, 480))
    cv2.imshow("Query", query_resized)

    for img_path in img_paths:
        result_img = cv2.imread(img_path)
        result_resized = cv2.resize(result_img, (720, 480))
        cv2.imshow("Result", result_resized)
        cv2.waitKey(0)


def write_time():
    row = [T_MODEL, T_FEAT_REDUCTION, T_SEARCH, T_ALL]
    save_to_csv("./experiments/holidays.csv", row)


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
        "-F",
        "--force",
        help="Choose the brute force search.",
        action="store_true",
    )
    argParser.add_argument(
        "-d",
        "--dataset",
        help="Path to where images are being stored"
    )
    args = vars(argParser.parse_args())

    if (args["force"]):
        brute_force_search(
            query_img_path=args["query"], dataset=args["dataset"])
    else:
        search(query_img_path=args["query"],
               cli=args["terminal"], dataset=args["dataset"])

    row = [T_MODEL, T_SEARCH, T_ALL]
    save_to_csv("../experiments/oxford.csv", row)
