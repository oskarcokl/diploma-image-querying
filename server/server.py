import tornado.ioloop
import tornado.web
import json
import logging
from tornado.httpclient import AsyncHTTPClient

from tornado.options import define, options, parse_command_line

define("port", default=8888, help="run on the given port", type=int)
define("debug", default=True, help="run in debug mode")


class MainHandler(tornado.web.RequestHandler):
    def get(self):
        self.write("Hello, world")


class FileUploadHandler(tornado.web.RequestHandler):
    def post(self):
        for field_name, files in self.request.files.items():
            for info in files:
                filename, content_type = info["filename"], info["content_type"]
                body = info["body"]
                logging.info(
                    'POST "%s" "%s" %d bytes', filename, content_type, len(body)
                )

        self.write("OK")


class JsonHandler(tornado.web.RequestHandler):
    async def get(self):
        response = await self.async_fetch("http://localhost:3000/images")
        json_response = tornado.escape.json_decode(response.body)
        self.write(json_response)

    async def async_fetch(self, url):
        http_client = AsyncHTTPClient()
        response = await http_client.fetch(url)
        return response


def main():
    parse_command_line()
    app = tornado.web.Application(
        [
            (r"/", MainHandler),
        ],
        debug=options.debug,
    )
    app.listen(options.port)
    tornado.ioloop.IOLoop.current().start()


if __name__ == "__main__":
    main()
