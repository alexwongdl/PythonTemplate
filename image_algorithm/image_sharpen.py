# coding: utf-8
"""
Created by Alex Wang
On 2017-10-30
image sharpen
"""
import os
import numpy as np
import cv2

from myutil import pathutil

def image_sharpening(img_org):
    kernel = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])
    im = cv2.filter2D(img_org, -1, kernel)
    return im

if __name__ == "__main__":
    dir = '/data/hzwangjian1/image_enhance/heibian_keyframe'
    save_root = '/data/hzwangjian1/sharpen'
    # dir = 'E://temp/videoDuplicate/img_1030'
    file_obs_list, file_list = pathutil.list_files(dir)
    for (abs_filename, filename) in zip(file_obs_list, file_list):
        if filename.endswith('jpg'):
            save_path = os.path.join(save_root, filename)
            img_org = cv2.imread(abs_filename)
            img_sharpen = image_sharpening(img_org)
            cv2.imwrite(save_path, img_sharpen)