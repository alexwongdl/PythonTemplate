# coding: utf-8
"""
Created by Alex Wang
On 2017-08-31
"""
import time
import requests
import json

import logging
import urllib.request
import traceback

from PIL import Image

logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)


def download_image_to_memory_with_retry(img_url, retry=5):
    """
    :param img_url:
    :param retry:
    :return:
    """
    try_time = 0
    image = None
    while try_time < retry and not image:
        image = download_image_to_memory(img_url)
        if not image:
            try_time += 1
            print('download image {} failed, retry:{}'.format(img_url, try_time))
            time.sleep(0.2)
    return image


def download_image_to_memory(img_url):
    """
    download image to memory with PIL Image format
    :param img_url:
    :return:
    """
    try:
        response = requests.get(img_url, timeout=600, stream=True)
        response.raw.decode_content = True
        image = Image.open(response.raw)

        if not response.ok:
            return None

        return image
    except Exception as e:
        print('download failed:', img_url)
        traceback.print_exc()
        return None


def image_download_with_retry(img_url, save_path, retry=5):
    """
    :param img_url:
    :param save_path:
    :param retry:
    :return:
    """
    try_time = 0
    succeed = False
    while try_time < retry and not succeed:
        succeed = image_download(img_url, save_path)
        if not succeed:
            try_time += 1
            print('download image {} failed, retry:{}'.format(img_url, try_time))
            time.sleep(0.2)
    return succeed


def image_download(url, save_path):
    """
    根据url下载图片并保存到save_path
    :param url:
    :param save_path:
    :return:
    """
    with open(save_path, 'wb') as handle:
        response = requests.get(url, timeout=60, stream=True)
        if not response.ok:
            return False

        for block in response.iter_content(1024):
            if not block:
                break
            handle.write(block)
        return True


def send_get(url, params=None):
    """
    发送get请求，params是dict格式
    :param url:
    :param params:
    :return:返回结果包含 status_code，headers，content属性，例如r.content来访问content属性
    """
    return requests.get(url, params)


def send_get_content(url, params=None):
    """
    发送get请求，params是dict格式
    :param url:
    :param params:
    :return:返回content
    """
    return requests.get(url, params).content.decode('UTF-8')


def send_post(url, params=None):
    """
    发送post请求，params是dict格式
    :param url:
    :param params:
    :return:返回结果包含 status_code，headers，content属性，例如r.content来访问content属性
    """
    return requests.post(url, params)


def send_post_content(url, params=None):
    """
    发送post请求，params是dict格式
    :param url:
    :param params:
    :return:返回content
    """
    return requests.post(url, params).content.decode('UTF-8')


def download_video(video_url, save_path):
    """
    下载视频
    python2:
        def download_video(video_url, save_path):
        rsp = urllib2.urlopen(video_url)
        with open(save_path, 'wb') as f:
            f.write(rsp.read())
    :param video_url:
    :param save_path: xxx.mp4
    :return:
    """
    urllib.request.urlretrieve(video_url, save_path)


if __name__ == "__main__":
    # image_download('http://dmr.nosdn.127.net/v-20170826-6b05cdaa733282703f729b5afcc65759.jpg','E://temp/docduplicate/image/v-20170826-6b05cdaa733282703f729b5afcc65759.jpg')

    # params = {'picUrl': 'http://img1.gtimg.com/20/2015/201558/20155894_980x1200_281.jpg'}
    # response = send_post_content('http://nlp.service.163.org/cv-api-logo/watermarker_detect', params)
    # print(response)

    title_cheat_url = 'http://nlp.service.163.org/dl-nlp-news/titlecheat_detect_article'
    params = {"title": "孙悟空被压五指山，菩提法师为啥不救他？背后原因吓人", "category": "人文"}
    response = send_post_content(title_cheat_url, params)
    print(response)
    response_json = json.loads(response)
    print(response_json['body']['finalMark'])

    sansu_url = 'http://nlp.service.163.org/news-api/vulgarity_quant_article'
    params = {"title": "美国怪兽车大赛 一名车手的表演燃爆全场", "category": "搞笑", 'docid': 'VCSAT9DI8', 'content': '',
              'source': '总有刁民想害朕'}
    response = send_post_content(sansu_url, params)
    print(response)
    response_json = json.loads(response)
    print(response_json['body']['serverity'])
