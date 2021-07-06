import argparse
import os


from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing import image
from tensorflow import keras
import numpy as np

import sys
sys.path.insert(0, "../")
sys.path.insert(0, "./")


from adder import Adder


def add_cli(img_list):
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

    img_name_list = []
    feature_list = []

    for img_path in img_list:
        img_name = img_path.split("/")[-1]
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        get_fc2_layer_output = K.function(
            [model.layers[0].input], model.layers[22].output
        )
        features = get_fc2_layer_output([img_array])[0]

        img_name_list.append(img_name)
        feature_list.append(features)

    adder = Adder()

    ids = []
    for i in range(len(img_name_list)):
        id = adder.add_img_to_db(feature_list[i], img_name_list[i])
        ids.append(id)

    print(ids)


if __name__ == "__main__":

    argParser = argparse.ArgumentParser()
    argParser.add_argument("-i", "--images", nargs="+", required=True,
                           help="Pass paths to images you want to add to index.")

    args = vars(argParser.parse_args())

    add_cli(args["images"])
