"""
Created by Alex.Wang
on 20170721

Tornado + Tensorflow 多进程失败
You may only use fork-safe libraries before entering multi-process mode (by calling http_server.start(N)).
Many complex libraries are not fork-safe.
You must either move the initialization of tensorflow.Session() after the HTTP server is started (in which case you will have 10 sessions, one in each process),
or you can create a separate tensorflow server and connect to it using the target argument of tensorflow.Session (sessions with a target are fork-safe).
The latter option is described in tensorflow/tensorflow#2448

my recommendation is to not use multi-process mode and always use an external process manager and load balancer instead. then you don't have to do anything special and everything will naturally be one-per-process
if you still want to use multi-process mode, just rearrange things so the call to HTTPServer.start comes before the creation of the tensorflow session
"""
import tornado.httpserver
import tornado.ioloop
import tornado.options
import tornado.web
from tornado.options import define
import os
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = ""
define("port", default=18015, help="run on the given port", type=int)


class IndexHandler(tornado.web.RequestHandler):
    def get(self):
        num = self.get_argument('num', 2)
        for i in range (1000):
            ret = self.application.calltf(num)
        self.write( 'result:' + str(ret))

class TFApplication(tornado.web.Application):
    def __init__(self):
        handlers=[(r"/", IndexHandler)]
        super(TFApplication, self).__init__(handlers)

        ## 初始化tensorflow
        self.a = tf.placeholder(tf.int32, shape=(), name="input")
        self.asquare = tf.multiply(self.a, self.a, name="output")
        config = tf.ConfigProto(device_count={"CPU": 20}, inter_op_parallelism_threads=10, intra_op_parallelism_threads=10)
        self.sess = tf.Session(config=config)
        print(self.sess.run([self.asquare], feed_dict={self.a: 32}))

    def calltf(self, num):
        print("num is:" + str(num))
        return self.sess.run([self.asquare], feed_dict={self.a: num})

if __name__ == "__main__":
    tornado.options.parse_command_line()
    http_server = tornado.httpserver.HTTPServer(TFApplication())

    http_server.bind(18825)
    http_server.start(2)
    tornado.ioloop.IOLoop.current().start()
