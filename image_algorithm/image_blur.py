# coding: utf-8
"""
Created by Alex Wang
On 2017-10-11
图像模糊化处理，下采样+高斯模糊

下采样：https://docs.scipy.org/doc/scipy-0.18.1/reference/generated/scipy.misc.imresize.html
高斯模糊：https://stackoverflow.com/questions/17595912/gaussian-smoothing-an-image-in-python
"""
import os

import numpy as np
import scipy.ndimage as ndimage
import scipy.misc
from multiprocessing import Pool

import pathutil

def blur_img(img_path):
    img = ndimage.imread(img_path)
    img = ndimage.gaussian_filter(img, sigma=(2, 2, 0), order=0)

    img_shape = img.shape
    width = img_shape[0]
    height = img_shape[1]
    img = scipy.misc.imresize(img, (int(width/2), int(height/2)))
    img = scipy.misc.imresize(img, (width, height), interp='bicubic')
    return img

def blur_one_img(path_tuple):
    image_path, save_path = path_tuple
    blurred_img = blur_img(image_path)
    scipy.misc.imsave(save_path, blurred_img)

def blur(input_dir, output_dir):
    print('start blur image')
    pool = Pool(15)
    path_tuple_list = []
    if not pathutil.dir_exist(output_dir):
        os.mkdir(output_dir)
    file_obs_list, file_list = pathutil.list_files(input_dir)
    for (abs_path, file_name) in zip(file_obs_list, file_list):
        # blurred_img = blur_img(abs_path)
        # scipy.misc.imsave(os.path.join(output_dir, file_name), blurred_img)
        path_tuple_list.append((abs_path, os.path.join(output_dir, file_name)))

    pool.map(blur_one_img, path_tuple_list)
    pool.close()
    pool.join()


if __name__ == "__main__":
    blur('E://temp/deblur/RAISE_HR_small','E://temp/deblur/RAISE_HR_small_blur')