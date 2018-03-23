"""
Created by Alex Wang on 2017-11-14
"""
import json
from alexutil import httputil
from alexutil import stringutil

def test_video_quality(video_url='http://kkkkkkk.mp4'):
    """
    视频静态质量测试
    :param video_url:
    :return:
    """
    params = dict({'video_url': video_url})
    response = httputil.send_post('http://localhost:19877/video_quality', params)
    print(stringutil.to_str(response.content))

    dict_ret = json.loads(stringutil.to_str(response.content))
    for key, value in dict_ret.items():
        print(key, value)

if __name__ == "__main__":
    test_video_quality()
