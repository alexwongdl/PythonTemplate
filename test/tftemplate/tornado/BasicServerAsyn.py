import tornado.httpserver
import tornado.ioloop
import tornado.options
import tornado.web
from tornado.options import define
from concurrent.futures import ThreadPoolExecutor
from tornado.concurrent import run_on_executor
import time

define("port", default=18015, help="run on the given port", type=int)

import tensorflow as tf
a = tf.placeholder(tf.int32, shape=(), name="input")
asquare = tf.multiply(a, a, name="output")
sess = tf.Session()

class IndexHandler(tornado.web.RequestHandler):
    executor = ThreadPoolExecutor(4)

    def initialize(self):
        self.sess = sess

    @tornado.web.asynchronous
    @tornado.gen.coroutine
    def get(self):
        greeting = self.get_argument('greeting', 'Hello')
        res = yield self.my_func()
        self.write(greeting + ', friendly user!' + str(res))

    # 线程里处理
    @run_on_executor
    def my_func(self):
        for i in range(10000):
            data = self.sess.run([asquare], feed_dict={a: 2})
        return data

if __name__ == "__main__":
    tornado.options.parse_command_line()
    app = tornado.web.Application(handlers=[(r"/", IndexHandler)])
    http_server = tornado.httpserver.HTTPServer(app)
    http_server.listen(18844)
    tornado.ioloop.IOLoop.current().start()

    #http_server.bind(18825)
    #http_server.start(3)
    #tornado.ioloop.IOLoop.current().start()
