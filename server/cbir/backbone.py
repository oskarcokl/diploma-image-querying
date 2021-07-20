import os

from tensorflow import keras
from keras.applications.vgg16 import VGG16
from keras.models import Model


class Backbone:
    """
    The class will be used for loading a CNN model and extracting
    features from images with said model. 
    """
    model = None

    def load_model(self):
        """
        Loads model either from disk if it is already downloaded or
        it downloads it from the internet.
        """
        if os.path.isdir("./vgg16"):
            model = self._load_model_from_disk()
        else:
            model = self._download_model()
            self._save_model_to_disk(model)

        self.model = Model(inputs=model.inputs,
                           outputs=model.layers[-2].output)

    def _load_model_from_disk(self):
        print("Model already downloaded loading from disk.")
        model = keras.models.load_model("./vgg16")
        return model

    def _download_model(self):
        print("Downloading model.")
        model = VGG16(
            weights="imagenet",
        )
        return model

    def _save_model_to_disk(self, model):
        print("Saving model to disk.")
        model.save("./vgg16")

    def get_features(self, img):
        """
        Function extracs features from preprocessed img and returns them.

        Parameters:
            img: np.ndarray

        """
        features = self.mode.predict(img)
        return features
