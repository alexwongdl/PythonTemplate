"""
Created by Alex Wang on 2018-06-01
"""

import os
import sys
import traceback
from collections import Counter
import time

import cv2
import numpy as np


def crop_image(img, debug=False, plot=False):
    """
    :param img:
    :param debug:
    :param plot:
    :return:
    """
    try:
        ratio = 4

        time_one = time.time()
        img_resize = cv2.resize(img, (0, 0), fx=1.0/ratio, fy=1.0/ratio)
        img_org = img.copy()
        img_gray = cv2.cvtColor(img_resize, cv2.COLOR_BGR2GRAY)
        ret, img_mask = cv2.threshold(img_gray, 240, 255, cv2.THRESH_BINARY)
        height, width = img_resize.shape[0:2]
        mask = np.zeros([height + 2, width + 2, 1], np.uint8)

        time_two = time.time()
        flood_y = int(height / 2)
        for x in range(50, width - 50, 10):
            # print(img_mask[y, flood_x])
            cv2.circle(img_mask, (x, flood_y), 2, 0, thickness=3)
            # print(img_mask[x, flood_y])
            if img_mask[flood_y, x] == 0:
                cv2.floodFill(img_mask, mask, (x, flood_y), 128, cv2.FLOODFILL_MASK_ONLY)

        time_three = time.time()
        gray_idx = np.argwhere(img_mask == 128)
        print(len(gray_idx))
        time_four = time.time()
        # print(gray_idx)
        x_idx, y_idx = zip(*gray_idx)
        # min_x = min(x_idx)
        # max_x = max(x_idx)
        # min_y = min(y_idx)
        # max_y = max(y_idx)
        x_counter = Counter(x_idx)
        x_counter_filter = {key: value for key, value in x_counter.items() if value > 10}
        y_counter = Counter(y_idx)
        y_counter_filter = {key: value for key, value in y_counter.items() if value > 10}

        min_x = min(x_counter_filter.keys())
        max_x = max(x_counter_filter.keys())
        min_y = min(y_counter_filter.keys())
        max_y = max(y_counter_filter.keys())
        img_new = img_org[min_x * ratio:max_x * ratio, min_y * ratio:max_y * ratio, :]
        if debug:
            print('img height:{}, img width:{}'.format(height, width))
            print('min_x:{}, max_x:{}, min_y:{}, max_y:{}'.format(min_x, max_x, min_y, max_y))

        time_five = time.time()
        height_threshold = int(height * 0.12)
        width_threshold = int(width * 0.1)
        if debug:
            print('threshold cost time:{}, flood fill cost time:{}, argwhere cost time:{}, '
                  'counter cost time:{}'.
                  format((time_two - time_one), (time_three - time_two),
                         (time_four - time_three), (time_five - time_four)))
            print('height_threshold:{}, width_threshold:{}'.format(height_threshold, width_threshold))

        if plot:
            cv2.imshow('img_org', img)
            cv2.imshow('img_mask', img_mask)
            cv2.imshow('img_new', img_new)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        if min_x > height_threshold or (height - max_x) > height_threshold \
                or min_y > width_threshold or (width - max_y) > width_threshold:
            if debug:
                print('white edge too large, return None.')
            return None, False

        return img_new, True
    except Exception as e:
        traceback.print_exc()
    return None, False


def test_batch(dir_path, debug=False, plot=False):
    """
    :param dir_path:
    :param debug:
    :param plot:
    :return:
    """
    save_dir = '/Users/alexwang/data/image_split/white_edge_crop_result'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    file_name_list = [
        'TB22IW5d3nH8KJjSspcXXb3QFXa_!!681671909.jpg',
        'TB2fsQWrASWBuNjSszdXXbeSpXa_!!1830643265.jpg',
        'TB2jmU1ruSSBuNjy0FlXXbBpVXa_!!3476998202.jpg',
        'TB2MaBIreSSBuNjy0FlXXbBpVXa_!!1768687087.jpg'
    ]

    # file_name_list = [
    #     'TB2dG9nfEOWBKNjSZKzXXXfWFXa_!!2564807339.jpg',
    #     'TB2E6TIdY5YBuNjSspoXXbeNFXa_!!678523567.jpg',
    #     'TB2fsQWrASWBuNjSszdXXbeSpXa_!!1830643265.jpg',
    #     'TB2FWYYrbGYBuNjy0FoXXciBFXa_!!1695755196.jpg',
    #     'TB2gmfmquuSBuNjSsplXXbe8pXa_!!1665834141.jpg',
    #     'TB2jmU1ruSSBuNjy0FlXXbBpVXa_!!3476998202.jpg',
    #     'TB2MaBIreSSBuNjy0FlXXbBpVXa_!!1768687087.jpg',
    #     'TB2mIBWfyMnBKNjSZFoXXbOSFXa_!!2624818026.jpg'
    # ]

    file_name_list = [
        'TB2.5kyppOWBuNjy0FiXXXFxVXa_!!3571500344.jpg',
        'TB2.XmpcHSYBuNjSspiXXXNzpXa_!!2244221894.jpg',
        'TB22uCGqHSYBuNjSspiXXXNzpXa_!!2871469722.jpg',
        'TB2_u5nouuSBuNjy1XcXXcYjFXa_!!2207560322.jpg',
        'TB2aNCCntqUQKJjSZFIXXcOkFXa_!!2655745701.jpg',
        'TB2ARccrDtYBeNjy1XdXXXXyVXa_!!2488474519.jpg',
        'TB2BWPqolHH8KJjy0FbXXcqlpXa_!!2964419525.jpg',
        'TB2ccv2X7v85uJjSZFPXXch4pXa_!!2917031043.jpg',
        'TB2CKu0rpuWBuNjSszbXXcS7FXa_!!2197260583.jpg',
        'TB2dG9nfEOWBKNjSZKzXXXfWFXa_!!2564807339.jpg',
        'TB2E6TIdY5YBuNjSspoXXbeNFXa_!!678523567.jpg',
        'TB2ebcNrbGYBuNjy0FoXXciBFXa_!!1974378418.jpg',
        'TB2fsQWrASWBuNjSszdXXbeSpXa_!!1830643265.jpg',
        'TB2FWYYrbGYBuNjy0FoXXciBFXa_!!1695755196.jpg',
        'TB2gmfmquuSBuNjSsplXXbe8pXa_!!1665834141.jpg',
        'TB2jmU1ruSSBuNjy0FlXXbBpVXa_!!3476998202.jpg',
        'TB2L1MpbiMnBKNjSZFCXXX0KFXa_!!3399312947.jpg',
        'TB2MaBIreSSBuNjy0FlXXbBpVXa_!!1768687087.jpg',
        'TB2mIBWfyMnBKNjSZFoXXbOSFXa_!!2624818026.jpg',
        'TB2NkeKi3mTBuNjy1XbXXaMrVXa_!!2676392390.jpg',
        'TB2Oa0XqStYBeNjSspaXXaOOFXa_!!495464744.jpg',
        'TB2VsYPisjI8KJjSsppXXXbyVXa_!!65866399.jpg',
        'TB2wN0BqQyWBuNjy0FpXXassXXa_!!2519883254.jpg',
        'TB283O3b_qWBKNjSZFxXXcpLpXa_!!3564256353.jpg'
    ]

    # for file_name in file_name_list:
    for file_name in os.listdir(dir_path):
        file_path = os.path.join(dir_path, file_name)
        print('file_path:', file_path)
        img = cv2.imread(file_path)
        start_time = time.time()
        img_result, succeed = crop_image(img, debug, plot)
        end_time = time.time()
        print('cost time:{}'.format(end_time - start_time))
        if succeed:
            cv2.imwrite(os.path.join(save_dir, file_name), img_result)


if __name__ == '__main__':
    test_batch('/Users/alexwang/data/image_split/white_edge_data', debug=True, plot=False)
