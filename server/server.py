import tornado.ioloop
import tornado.web
import json
import logging
import cv2
import numpy as np
from celery_app.tasks import add
from celery_app.cbir_tasks import cbir_query
from tornado.options import define, options, parse_command_line

define("port", default=8888, help="run on the given port", type=int)
define("debug", default=True, help="run in debug mode")


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


class FileUploadHandler(BaseHandler):
    def post(self):
        for field_name, files in self.request.files.items():
            for info in files:
                filename, content_type = info["filename"], info["content_type"]
                body = info["body"]
                logging.info(
                    'POST "%s" "%s" %d bytes', filename, content_type, len(
                        body)
                )

        self.write("OK")


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
        for field_name, files in self.request.files.items():
            decoded_img_array = decode_uploaded_img(files)
            decoded_img_list = decoded_img_array.tolist()
            result_imgs = cbir_query.delay(
                query_img_list=decoded_img_list, cli=False
            ).get()
            result = {"result_imgs": result_imgs}
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
            (r"/file-upload", FileUploadHandler),
            (r"/celery", CeleryHandler),
            (r"/cbir-query", CBIRQueryHandler),
        ],
        debug=options.debug,
    )
    app.listen(options.port)
    server_startup_message()
    tornado.ioloop.IOLoop.current().start()


def server_startup_message():
    print(f"Tornado server is running on port {options.port}")
    print(f"Connect to server from url: http://localhost:{options.port}/")


def decode_uploaded_img(file):
    img_bytes = file[0].body
    img_array = cv2.imdecode(np.frombuffer(
        img_bytes, np.uint8), cv2.IMREAD_COLOR)
    img_array = cv2.resize(img_array, (224, 224))
    img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    # Think image should be RGB after this but couldn't test.
    # Chechk here if some problem arise.
    return img_array


if __name__ == "__main__":
    main()
