from re import A
from numpy.core.defchararray import decode
import tornado.ioloop
import tornado.web
import json
import os
import logging
import cv2
import numpy as np
from tornado.options import define, options, parse_command_line
from tensorflow.keras.applications.resnet import preprocess_input

from db_utils.table_operations import get_feature_vectors
from cbir.backbone import Backbone
from rocchio import make_new_query
from celery_app.cbir_tasks import cbir_query, index_add
from celery_app.tasks import add

define("port", default=8888, help="run on the given port", type=int)
define("debug", default=True, help="run in debug mode")


backbone = Backbone()
print("Loaded backbone")

n_images = 10

# Coppied from https://stackoverflow.com/questions/35254742/tornado-server-enable-cors-requests
# Thank you kwarunek :^)


class BaseHandler(tornado.web.RequestHandler):
    def set_default_headers(self):
        print("setting headers!!!")
        self.set_header("Access-Control-Allow-Origin", "*")
        self.set_header("Access-Control-Allow-Headers", "x-requested-with")
        self.set_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")

    def post(self):
        self.write("some post")

    def get(self):
        self.write("some get")

    def options(self):
        # no body
        self.set_status(204)
        self.finish()


class MainHandler(BaseHandler):
    def get(self):
        self.write("Hello, world")


class AddIndexHandler(BaseHandler):
    def post(self):
        for field_name, files in self.request.files.items():
            decoded_images = []
            image_names = []
            for info in files:
                body = info["body"]

                path_to_save = os.path.join("./test", info["filename"])
                with open(path_to_save, "wb") as f:
                    f.write(body)

                decoded_image = decode_uploaded_img(body)
                decoded_images.append(
                    (info["filename"], decoded_image.tolist()))

            add = index_add.delay(decoded_images).get()

        if (add):
            self.write("ok")
        else:
            self.write("notok")


class ROCCHIOQueryHandler(BaseHandler):
    def set_default_headers(self):
        self.set_header("Content-Type", "application/json")
        self.set_header("Access-Control-Allow-Origin", "*")

    def post(self):
        result_json = None
        rocchio_data_str = (self.get_body_argument(
            "rocchioData", default=None, strip=False))
        rocchio_data = json.loads(rocchio_data_str)

        old_query_feautres = np.array(rocchio_data["query"])
        relevant_images = []
        non_relevant_images = []

        images = rocchio_data["selectedImages"]

        for image_name in images:
            image_data = images[image_name]
            if image_data["selected"]:
                relevant_images.append(np.array(image_data["feature_vector"]))
            else:
                non_relevant_images.append(
                    np.array(image_data["feature_vector"]))

        new_query = make_new_query(
            old_query_feautres, relevant_images, non_relevant_images)

        new_query_feautres_list = new_query.tolist()

        result_imgs = cbir_query.delay(
            cli=False, query_features=new_query_feautres_list, n_images=10
        ).get()

        result = {"ordered_result": result_imgs,
                  "dict": {}, "query_features": new_query_feautres_list}
        feautre_vectors = get_feature_vectors(result_imgs)

        for index, result_img in enumerate(result_imgs):
            name = result_img.split(".")[0]
            result["dict"][result_img] = {
                "name": name, "feature_vector": feautre_vectors[index], "selected": False}

        result_json = json.dumps(result)
        self.write(result_json)


class CBIRQueryHandler(BaseHandler):
    def set_default_headers(self):
        self.set_header("Content-Type", "application/json")
        self.set_header("Access-Control-Allow-Origin", "*")

    async def get(self):
        results = cbir_query.delay().get()
        for result in results:
            print(result)
        result_str = ",".join(results)

        self.write(result_str)

    def post(self):
        result_json = None
        selected_images_str = (self.get_body_argument(
            "selectedImages", default=None, strip=False))
        selected_images = json.loads(selected_images_str)

        print(selected_images)

        for field_name, files in self.request.files.items():
            decoded_img_array = decode_uploaded_img(files[0].body)
            query_features = backbone.get_features(decoded_img_array)
            query_features_list = query_features.tolist()
            result_imgs = cbir_query.delay(
                cli=False, query_features=query_features_list, n_images=10
            ).get()

            result = {"ordered_result": result_imgs,
                      "dict": {}, "query_features": query_features_list}
            feautre_vectors = get_feature_vectors(result_imgs)

            for index, result_img in enumerate(result_imgs):
                name = result_img.split(".")[0]
                result["dict"][result_img] = {
                    "name": name, "feature_vector": feautre_vectors[index], "selected": False}

            result_json = json.dumps(result)
        self.write(result_json)


class CeleryHandler(BaseHandler):
    async def get(self):
        result = add.delay(3, 4).get()
        print(str(result))
        self.write(str(result))


def main():
    parse_command_line()
    app = tornado.web.Application(
        [
            (r"/", MainHandler),
            (r"/add-index", AddIndexHandler),
            (r"/celery", CeleryHandler),
            (r"/cbir-query", CBIRQueryHandler),
            (r"/rocchio-query", ROCCHIOQueryHandler),
        ],
        debug=options.debug,
    )
    app.listen(options.port)
    server_startup_message()
    tornado.ioloop.IOLoop.current().start()


def server_startup_message():
    print(f"Tornado server is running on port {options.port}")
    print(f"Connect to server from url: http://localhost:{options.port}/")


def decode_uploaded_img(img_bytes):
    img_array = cv2.imdecode(np.frombuffer(
        img_bytes, np.uint8), cv2.IMREAD_COLOR)
    img_array = cv2.resize(img_array, (224, 224))
    img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    np_img_array = np.array(img_array)
    np_img_array = np.expand_dims(np_img_array, axis=0)
    processed_img_array = preprocess_input(np_img_array)
    # Think image should be RGB after this but couldn't test.
    # Chechk here if some problem arise.
    return processed_img_array


if __name__ == "__main__":
    main()
