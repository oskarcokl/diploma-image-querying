import argparse
import os
import sys

import psycopg2
import numpy as np
from tensorflow import keras
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras import backend as K

from config import config
from create_table import create_table


# Local application imports
sys.path.insert(0, "../")
from db_connector import DbConnector
from models.cd_tree import CDTree


def insert_image_vector(image_name, image_vector):
    sql = """INSERT INTO cbir_index(image_name, image_vector)
             VALUES(%s, %s) RETURNING id;"""

    connection = None
    image_id = None
    try:
        params = config()
        connection = psycopg2.connect(**params)
        cursor = connection.cursor()
        cursor.execute(sql, (image_name, image_vector))
        image_id = cursor.fetchone()[0]
        connection.commit()
        cursor.close()
    except (Exception, psycopg2.DatabaseError) as e:
        print(e)
    finally:
        if connection is not None:
            connection.close()

    return image_id


def insert_image_vector_list(tuple_list):
    sql = """INSERT INTO cbir_index(image_name, image_vector)
             VALUES(%s, %s) RETURNING id;"""

    connection = None
    image_id = None
    try:
        print("Writing image vectors to database.")
        params = config()
        connection = psycopg2.connect(**params)
        cursor = connection.cursor()
        cursor.executemany(sql, tuple_list)
        connection.commit()
        cursor.close()
    except (Exception, psycopg2.DatabaseError) as e:
        print(e)
    finally:
        if connection is not None:
            connection.close()
            print("Finished writing image vectors to database.")

    return image_id


def index():

    pass


# This function is intented to be run only when setting up the initial db.
def init_index(dataset_src):
    commands = (
        """
        CREATE TABLE cbir_index (
            id SERIAL PRIMARY KEY,
            image_name VARCHAR(255) NOT NULL,
            image_vector DOUBLE PRECISION[4096]
        )
        """,
    )
    create_table(commands)

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

    img_path_list = []
    feature_list = []

    for img_name in os.listdir(dataset_src):
        img_path = os.path.join(dataset_src, img_name)
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        get_fc2_layer_output = K.function(
            [model.layers[0].input], model.layers[21].output
        )
        features = get_fc2_layer_output([img_array])[0]
        features_to_list = features.tolist()

        img_path_list.append(img_path)
        feature_list.append(features_to_list)

    tuple_list = list(zip(img_path_list, feature_list))

    insert_image_vector_list(tuple_list)


def init_cd_tree(data, min_clusters, max_clusters, min_node, l_max):

    cd_tree = CDTree(min_node, l_max)

    root_node = cd_tree.init_cd_tree(data, min_clusters, max_clusters)
    pass


def get_data():
    connector = DbConnector()
    connector.cursor.execute("SELECT * FROM cbir_index")
    print("Number of indexed images: ", connector.cursor.rowcount)
    data = connector.cursor.fetchall()
    data_array = np.array(data, dtype=object)

    rand_indexes = np.random.choice(
        1909, 10, replace=False
    )
    print(rand_indexes)
    rand_data = data_array[rand_indexes]
    print(f"Lenght of subset of data {len(rand_data)}")
    return rand_data


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
        help="Set if initializing the db for the first time. Will add all pictures in dataset path",
        action="store_true",
    )
    argParser.add_argument(
        "-IT",
        "--init-cd-tree",
        help="Set if initializing the cd tree for the first time. Will create a cd tree and store it with ZODB.",
        action="store_true",
    )
    args = vars(argParser.parse_args())
    if args.get("init-db"):
        init_index(args.get("dataset"))
    elif args.get("init_cd_tree"):
        data = get_data()
        init_cd_tree(data, 2, 4, 15, 5)
