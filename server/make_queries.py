import argparse
import os
import numpy as np
from cbir.searcher import Searcher
from sklearn.decomposition import TruncatedSVD
from cbir.backbone import Backbone
from db_utils.db_connector import DbConnector
from csv_writer import init_csv


from cbir import search


holidays = "../../dataset/vacations"
objects = "../../dataset/256_objects"
coco = "../../../../Datasets/train2017"


def make_queries(file_name, force=False):
    result_lines = []
    dataset_name = "coco.csv"

    init_csv(os.path.join("experiments/", dataset_name))

    with open(file_name, "r") as f:
        lines = f.readlines()
        backbone = Backbone()

        if not force:
            feature_vectors = get_feature_vectors()
        else:
            img_names, feature_vectors = get_names_and_features()
            #svd = TruncatedSVD(n_components=140)
            #svd.fit(feature_vectors)
            #reduced_feature_vectors = svd.transform(feature_vectors)
            reduced_feature_vectors = feature_vectors

        for line in lines:
            query_img_name = line[:-1]
            query_img_path = os.path.join(
                coco, query_img_name)

            if not force:
                result = search.search(query_img_path=query_img_path,
                                       cli=True, backbone=backbone,
                                       n_images=10,
                                       feature_vectors=feature_vectors)
            else:
                result = search.brute_force_search(
                    query_img_path=query_img_path,
                    backbone=backbone, n_images=10,
                    feature_vectors=feature_vectors,
                    img_names=img_names,
                    reduced_feature_vectors=reduced_feature_vectors)

            result_line = " ".join((query_img_name, result))
            print(result_line)
            result_lines.append(result_line)

    with open("results.dat", "w") as f:
        for result_line in result_lines:
            result_line = "".join((result_line, "\n"))
            f.write(result_line)


def get_feature_vectors():
    connector = DbConnector()
    connector.cursor.execute("SELECT * FROM cbir_index")
    data = connector.cursor.fetchall()
    data_array = np.array(data, dtype=object)

    feature_vectors = data_array[:, 2]
    result = np.array([np.array(feature_vector)
                       for feature_vector in feature_vectors])
    return result


def get_names_and_features():
    connector = DbConnector()
    connector.cursor.execute("SELECT * FROM cbir_index")
    data = connector.cursor.fetchmany(1000)
    data_array = np.array(data, dtype=object)

    img_names = data_array[:, 1]
    result_img_names = [np.array(img_name) for img_name in img_names]

    feature_vectors = data_array[:, 2]
    result_feature_vectors = [np.array(feature_vector)
                              for feature_vector in feature_vectors]
    return np.array(result_img_names), np.array(result_feature_vectors)


if __name__ == "__main__":
    argParser = argparse.ArgumentParser()
    argParser.add_argument(
        "-f",
        "--file",
        help="Path to file containing query names in columns."
    )
    argParser.add_argument(
        "-FR",
        "--force",
        help="Weather set this flag if you want to use the brute force algorithm",
        action="store_true"
    )

    args = vars(argParser.parse_args())

    make_queries(args["file"], args["force"])
