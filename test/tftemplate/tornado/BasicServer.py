import tornado.httpserver
import tornado.ioloop
import tornado.options
import tornado.web
from tornado.options import define

define("port", default=18015, help="run on the given port", type=int)

import tensorflow as tf
a = tf.placeholder(tf.int32, shape=(), name="input")
asquare = tf.multiply(a, a, name="output")
sess = tf.Session()

class IndexHandler(tornado.web.RequestHandler):
    def initialize(self):
        self.sess = sess

    def get(self):
        greeting = self.get_argument('greeting', 'Hello')
        print(self.sess.run([asquare], feed_dict={a: 2}))
        self.write(greeting + ', friendly user!')

if __name__ == "__main__":
    tornado.options.parse_command_line()
    app = tornado.web.Application(handlers=[(r"/", IndexHandler)])
    http_server = tornado.httpserver.HTTPServer(app)
    http_server.listen(18824)
    tornado.ioloop.IOLoop.current().start()

    #http_server.bind(18825)
    #http_server.start(3)
    #tornado.ioloop.IOLoop.current().start()
