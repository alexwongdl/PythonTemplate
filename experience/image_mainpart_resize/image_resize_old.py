"""
Created by Alex Wang
on 2018-05-14
"""

import os
import traceback

import numpy as np
import cv2


def split_and_resize(img, debug=False, plot=False):
    """
    get mainpart of image and resize to appropriate size and generate a new image of size 800*800
    :param img:
    :param debug:
    :param plot:
    :return:
    """
    if debug:
        print('shape of image:', img.shape)

    alpha_channel = img[:, :, 3]
    non_zero_x_axis, non_zero_y_axis = np.nonzero(alpha_channel)
    top_boundary = min(non_zero_x_axis)
    bottom_boundary = max(non_zero_x_axis)
    left_boundary = min(non_zero_y_axis)
    right_boundary = max(non_zero_y_axis)

    img_bgr = img[:, :, 0:3].copy()
    print(img_bgr.shape)
    zero_idx = np.argwhere(alpha_channel == 0)
    for (x, y) in zero_idx:
        img_bgr[x, y, :] = [255, 255, 255]

    img_boundary = img[top_boundary:bottom_boundary + 1,
                   left_boundary:right_boundary,
                   0:3].copy()

    if plot:
        cv2.imshow('img', img)
        cv2.imshow('img_bgr', img_bgr)
        cv2.imshow('img_boundary', img_boundary)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def process_batch(dir_path):
    """
    :param dir_path:
    :return:
    """
    for file_name in os.listdir(dir_path):
        file_path = os.path.join(dir_path, file_name)
        print(file_path)
        # file_path = os.path.join(dir_path, 'LB1N.wsXd3XBuNjt_n_XXcDSpXa.png_539.png')
        try:
            if os.path.isfile(file_path):
                img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
                print(img[:, :, 3])
                split_and_resize(img, debug=True, plot=True)

        except Exception as e:
            traceback.print_exc()


if __name__ == '__main__':
    print('test...')
    process_batch('/Users/alexwang/data/image_resize/image_seg')
