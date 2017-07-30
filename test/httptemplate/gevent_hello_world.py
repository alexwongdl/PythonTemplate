"""
Created by Alex Wang
on 2017-07-30
"""

from gevent import monkey
monkey.patch_all()
from flask import Flask
from gevent import wsgi

app = Flask(__name__)

@app.route('/')
def index():
    return 'Hello World'

server = wsgi.WSGIServer(('127.0.0.1', 19876), app)
server.serve_forever()