# coding=utf-8
"""
Created by hzwangjian1 on 2017-11-14
提供视频静态质量服务接口
crontab定时创建目录和删除历史数据

需要修改video_feature_extractor.py
    data_dir = '/home/nlp/hzwangjian1/video_quality_service/data' if os.name != 'nt' else 'E:/data/input/news/video/'
"""

from gevent import monkey

monkey.patch_all()
from flask import Flask, request
from gevent import wsgi
import json

import video_quality_online
from alexutil import dateutil

app = Flask(__name__)


@app.route('/')
def index():
    return 'Hello World'


@app.route('/hello')
def response_request():
    num = request.args.get('num')
    ret = int(num) * int(num)
    return str(ret)

@app.route('/health')
def health_check():
    return 'ok'

@app.route('/cv-video/quality', methods=['POST'])
def call_video_quality_one():
    if request.method == 'POST':
        data_dict = request.form.to_dict()
        video_url = data_dict.get('video_url', '')

        display_image_url = ''
        if 'display_image_url' in data_dict.keys():
            display_image_url = data_dict.get('display_image_url', '')
        big_image_url = data_dict.get('big_image_url', '')
        category = data_dict.get('category', '')
        tid_level = data_dict.get('tid_level', '')
        day_str = dateutil.current_day_format()
        print(video_url)
        result = video_quality_online.process_video_single_online(video_url,
                                                                  non_value_process(big_image_url),
                                                                  non_value_process(category),
                                                                  non_value_process(tid_level), day_str, display_image_url)
        result.update({'quality_score':round(result['total_score'], 3), 'quality_level':result['total_score_level']})
        print(result)
        return json.dumps(result)
    else:
        content = json.dumps({"error_code": "1001"})
        return content

@app.route('/video_quality', methods=['POST'])
def call_video_quality():
    if request.method == 'POST':
        data_dict = request.form.to_dict()
        video_url = data_dict.get('video_url', '')

        display_image_url = ''
        if 'display_image_url' in data_dict.keys():
            display_image_url = data_dict.get('display_image_url', '')
        big_image_url = data_dict.get('big_image_url', '')
        category = data_dict.get('category', '')
        tid_level = data_dict.get('tid_level', '')
        day_str = dateutil.current_day_format()
        print(video_url)
        result = video_quality_online.process_video_single_online(video_url,
                                                                  non_value_process(big_image_url),
                                                                  non_value_process(category),
                                                                  non_value_process(tid_level), day_str, display_image_url)
        result.update({'quality_score':round(result['total_score'], 3), 'quality_level':result['total_score_level']})
        print(result)
        return json.dumps(result)
    else:
        content = json.dumps({"error_code": "1001"})
        return content


def non_value_process(attr):
    if attr is None or attr == "":
        return "  "
    else:
        return attr


if __name__ == "__main__":
    server = wsgi.WSGIServer(('127.0.0.1', 19898), app)
    server.serve_forever()
