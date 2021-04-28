import tornado.ioloop
import tornado.web

from tornado.options import define, options, parse_command_line

define("port", default=8888, help="run on the given port", type=int)
define("debug", default=True, help="run in debug mode")

class MainHandler(tornado.web.RequestHandler):
    def get(self):
        self.write("Hello, world")

class HelloHandler(tornado.web.RequestHandler):
    def get(self):
        self.write("Hello You made it to a different route")  


def main():
    parse_command_line()
    app = tornado.web.Application([
        (r"/", MainHandler),
        (r"/hello", HelloHandler),
    ])
    app.listen(options.port)
    tornado.ioloop.IOLoop.current().start()

if __name__ == "__main__":
    main()