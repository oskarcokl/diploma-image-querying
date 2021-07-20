import os

from tensorflow import keras
from tensorflow.keras.applications.vgg16 import VGG16


class Backbone:
    model = None

    def get_model(self):
        if os.path.isdir("./vgg16"):
            self.model = self._load_model_from_disk()
        else:
            self.model = self._download_model()
            self._save_model_to_disk(self, self.model)
        return self.model

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
