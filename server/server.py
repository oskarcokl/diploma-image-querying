import tornado.ioloop
import tornado.web
import json
from tornado.httpclient import AsyncHTTPClient

from tornado.options import define, options, parse_command_line

define("port", default=8888, help="run on the given port", type=int)
define("debug", default=True, help="run in debug mode")


class MainHandler(tornado.web.RequestHandler):
    def get(self):
        self.write("Hello, world")


class HelloHandler(tornado.web.RequestHandler):
    def get(self):
        self.write("Hello You made it to a different route")


class JsonHandler(tornado.web.RequestHandler):
    async def get(self):
        response = await self.async_fetch("http://localhost:3000/images")
        print(type(json.loads(response)[0]))
        self.write("Just testing some things :^)")

    async def async_fetch(self, url):
        http_client = AsyncHTTPClient()
        response = await http_client.fetch(url)
        string_value = response.body.decode("utf-8")
        return string_value


def main():
    parse_command_line()
    app = tornado.web.Application(
        [
            (r"/", MainHandler),
            (r"/hello", HelloHandler),
            (r"/json", JsonHandler),
        ],
        debug=options.debug,
    )
    app.listen(options.port)
    tornado.ioloop.IOLoop.current().start()


if __name__ == "__main__":
    main()
