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


def search(query_img_path, result_dir, cli):
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

    img = image.load_img(query_img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    get_fc2_layer_output = K.function([model.layers[0].input], model.layers[21].output)
    features_query = get_fc2_layer_output([img_array])[0]

    (dist, img_paths) = searcher.search(features_query.reshape(1, -1), 10)

    if cli:
        show_resutls(query_img_path, img_paths)
    else:
        return img_paths


def show_resutls(query_img_path, img_paths):
    query_img = cv2.imread(query_img_path)
    query_resized = cv2.resize(query_img, (720, 480))
    cv2.imshow("Query", query_resized)
    cv2.waitKey(0)

    for img_path in img_paths:
        result_img = cv2.imread(img_path)
        result_resized = cv2.resize(result_img, (720, 480))
        cv2.imshow("Result", result_resized)
        cv2.waitKey(0)


if __name__ == "__main__":
    argParser = argparse.ArgumentParser()
    argParser.add_argument(
        "-q", "--query", required=True, help="Path to the query image"
    )
    argParser.add_argument(
        "-r", "--result_dir", required=True, help="Path to results directory"
    )
    args = vars(argParser.parse_args())

    search(args["query"], args["result_dir"])
