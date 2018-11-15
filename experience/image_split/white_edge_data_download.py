#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created by Alex Wang on 2018-04-25
download split images from tfs

Data prepare:
    tunnel download -fd '\t' -limit 1000 search_offline.image_split_result_alexwang /Users/alexwang/data/image_split/image_split_alexwang.txt;
"""

import os
import random
from multiprocessing.dummy import Pool as ThreadPool
import json
import traceback

import shutil
from util.download_tfs_image import image_download_tuple


def parse_and_download_images():
    """
    :return:
    """
    file_name = '/Users/alexwang/data/image_split/white_edge_data_list.txt'
    save_dir = '/Users/alexwang/data/image_split/white_edge_data'

    if os.path.isdir(save_dir):
        # os.remove(save_dir)
        shutil.rmtree(save_dir)
    os.makedirs(save_dir)

    index = 0
    url_path_tuple_list = []
    for line in open(file_name, 'r'):
        try:
            elems = line.split('\t')
            url_str = elems[1].strip()
            judge_str = elems[2].strip()

            url_json = json.loads(url_str)
            judge_json = json.loads(judge_str)

            url = url_json['url']
            opt_1 = judge_json[0]['option']
            opt_2 = ''
            if 'option' in judge_json[1]:
                opt_2 = judge_json[1]['option']

            url_utf_8 = url.encode('utf-8')
            print('url:', url_utf_8)
            if u'\u526a\u88c1\u4e0d\u5408\u7406\uff0c\u7559\u767d\u8fb9' in opt_2:
                print('judge opt_1:', opt_1)
                print('judge opt_2:', opt_2)
                url_path_tuple_list.append((url_utf_8, os.path.join(save_dir, os.path.basename(url_utf_8))))
            index += 1
        except Exception as e:
            traceback.print_exc()

    pool = ThreadPool(100)
    pool.map(image_download_tuple, url_path_tuple_list)
    pool.close()
    pool.join()


if __name__ == '__main__':
    parse_and_download_images()

# \u7b26\u5408: 符合
# \u4e0d\u7b26\u5408: 不符合
#