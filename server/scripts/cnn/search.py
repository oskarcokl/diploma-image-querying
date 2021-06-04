import argparse
import os
from searcher import Search
from tensorflow import keras
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input


def search():
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

    print(model.summary())


if __name__ == "__main__":
    argParser = argparse.ArgumentParser()
    argParser.add_argument(
        "-q", "--query", required=True, help="Path to the query image"
    )
    argParser.add_argument(
        "-r", "--result_path", required=True, help="Path to results directory"
    )
    args = vars(argParser.parse_args())

    print(args)

    search()
