"""
Created by Alex Wang
on 2017-07-30
非阻塞异步flask服务
logger：https://stackoverflow.com/questions/26578733/why-is-flask-application-not-creating-any-logs-when-hosted-by-gunicorn
"""
import os
from gevent import monkey
monkey.patch_all()
from flask import Flask, request
from gevent import wsgi
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = ""
a = tf.placeholder(tf.int32, shape=(), name="input")
asquare = tf.multiply(a, a, name="output")
sess = tf.Session()

app = Flask(__name__)

@app.route('/')
def index():
    return 'Hello World'

@app.route('/hello')
def response_request():
    num = request.args.get('num')
    for i in range (100):
        ret = sess.run([asquare], feed_dict={a: num})
    return str(ret)
    # return "hello"

if __name__ == "__main__":
    server = wsgi.WSGIServer(('127.0.0.1', 19877), app)
    server.serve_forever()
