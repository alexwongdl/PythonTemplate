"""
Created by Alex Wang
on 2017-07-26
"""

import os

from flask import Flask, request
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = ""
a = tf.placeholder(tf.int32, shape=(), name="input")
asquare = tf.multiply(a, a, name="output")
sess = tf.Session()

app = Flask(__name__)
@app.route('/')
def hello_world():
    return "Hellow World"

def response_request():
    num = request.args.get('num')
    for i in range (100):
        ret = sess.run([asquare], feed_dict={a: num})
    return str(ret)
    # return "hello"


if __name__ == "__main__":
    app.add_url_rule("/hello", view_func=response_request)
    app.run(host='127.0.0.1',port=18997, debug=True)