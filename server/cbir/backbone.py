import logging
import os

from keras.applications.resnet import ResNet101
from keras.models import Model
from tensorflow import keras


class Backbone:
    """
    The class will be used for loading a CNN model and extracting
    features from images with said model. 
    """
    model = None
    model_dir = "./resnet101"

    def __init__(self):
        self.model = self.load_model()

    def load_model(self):
        """
        Loads model either from disk if it is already downloaded or
        it downloads it from the internet.
        """
        if os.path.isdir(self.model_dir):
            model = self._load_model_from_disk()
        else:
            model = self._download_model()
            self._save_model_to_disk(model)

        return Model(inputs=model.inputs,
                     outputs=model.layers[-2].output)

    def _load_model_from_disk(self):
        logging.info("Model already downloaded loading from disk.")
        model = keras.models.load_model(self.model_dir)
        return model

    def _download_model(self):
        logging.info("Downloading model.")
        model = ResNet101(
            weights="imagenet",
        )
        return model

    def _save_model_to_disk(self, model):
        logging.info("Saving model to disk.")
        model.save(self.model_dir)

    def get_features(self, img):
        """
        Function extracs features from preprocessed img and returns them.
        Before calling this function load_model must be called.

        Parameters:
            img: np.ndarray

        """
        if self.model is not None:
            features = self.model.predict(img)

        else:
            raise RuntimeError(
                "Model not loaded. Use load_model() to load it.")

        return features[0]
