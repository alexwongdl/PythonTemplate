"""
Created by Alex Wang on 2018-04-19
Use Hough line detect algorithm to split image
"""

import os
import math
import time
import traceback
from collections import Counter

import numpy as np
import scipy.stats
import cv2


def merge_group(group, width, debug):
    """
    merge lines in each group
    :param group:
    :param width:
    :param debug:
    :return:
    """
    flag_arr = np.zeros([width], dtype=np.int16)
    y_axis = group[0][1]
    count = 0
    for line in group:
        for i in range(line[0], line[2] + 1):
            flag_arr[i] |= 1
            # y_axis += line[1]
            # count += 1
    # y_axis = int(y_axis/count)

    length = np.sum(flag_arr)
    if debug:
        print('lines_group:')
        print(group)
        print('length:', length)

    return length, y_axis


def cal_patches_laplacian(img, y_axis_list, debug):
    """
    cal patches laplacian value to filt dump patches
    :param img:
    :param y_axis_list:
    :param debug:
    :return:
    """

    laplacian_val_threshold = 400
    height, width = img.shape[0:2]
    img_patches = []
    for i in range(len(y_axis_list) - 1):
        y_min = y_axis_list[i]
        y_max = y_axis_list[i + 1]
        if y_max - y_min <= 20:  # filter small patches
            continue

        img_temp = img[y_min:y_max, :, :]
        if debug:
            print('shape of img_temp laplacian:', img_temp.shape)

        laplacian_val = cv2.Laplacian(img_temp, cv2.CV_64F).var()
        if laplacian_val >= laplacian_val_threshold:
            img_patches.append(img_temp)

        if debug:
            print('y_min:', y_min, 'y_max:', y_max)
            print('laplacian_val', laplacian_val)

    return img_patches


def cal_patches_entropy(img_gray, img_org, y_axis_list, debug):
    """
    cal patches entropy and white pixel ratio
    :param img_gray:
    :param img_org:
    :param y_axis_list:
    :param debug:
    :return:
    """
    height, width = img_org.shape[0:2]
    entropy_threshold = 3
    white_ratio_threshold = 0.85

    img_patches = []
    entropy_list = []
    white_ratio_list = []

    y_split_list = [0]

    for i in range(len(y_axis_list) - 1):
        y_min = y_axis_list[i]
        y_max = y_axis_list[i + 1]
        # if y_max - y_min <= 20:  # filter small patches
        #     continue

        img_temp = img_gray[y_min:y_max, :]
        if debug:
            print('shape of img_temp entropy:', img_temp.shape)

        hist = cv2.calcHist([img_temp], [0], None, [256], [0, 256])
        hist = np.reshape(hist, newshape=(256))
        hist_prob = hist / (np.sum(hist) + math.exp(-6)) + math.exp(-10)
        entropy = scipy.stats.entropy(hist_prob)

        entropy_list.append(entropy)

        white_ratio = np.sum(img_gray[y_min + 1:y_max - 1, :] >= 250) / \
                      ((y_max - y_min - 2) * width + math.exp(-6))
        white_ratio_list.append(white_ratio)

        if debug:
            print('y_min:', y_min, 'y_max:', y_max)
            print('image entropy val:', entropy)
            print('image white pixel ratio:', white_ratio)
            print('')

    for i in range(1, len(y_axis_list) - 1):
        # if debug:
        #     print('{}/{}, y_axis_list:{}'.format(i, len(entropy_list), len(y_axis_list)))
        prev_entropy = entropy_list[i - 1]
        curr_entropy = entropy_list[i]
        prev_white_ratio = white_ratio_list[i - 1]
        curr_white_ratio = white_ratio_list[i]

        if (prev_entropy < entropy_threshold and curr_entropy >= entropy_threshold) or \
                (prev_entropy >= entropy_threshold and curr_entropy < entropy_threshold):
            if (prev_white_ratio < white_ratio_threshold and
                        curr_white_ratio >= white_ratio_threshold) or \
                    (prev_white_ratio >= white_ratio_threshold and
                             curr_white_ratio < white_ratio_threshold):
                y_split_list.append(y_axis_list[i])

    y_split_list.append(y_axis_list[len(y_axis_list) - 1])  # append image_height
    if (len(y_split_list) <= 2):
        return img_patches

    for i in range(len(y_split_list) - 1):
        y_min = y_split_list[i]
        y_max = y_split_list[i + 1]
        if (y_max - y_min) * 1.0 / width < 0.3:
            if debug:
                print('(y_max - y_min)/width < 0.3:{},{},{}'.format(y_max, y_min, width))
            continue

        if y_max - y_min > 20:
            img_temp = img_gray[y_min:y_max, :]
            hist = cv2.calcHist([img_temp], [0], None, [256], [0, 256])
            hist = np.reshape(hist, newshape=(256))
            hist_prob = hist / (np.sum(hist) + math.exp(-6))
            entropy = scipy.stats.entropy(hist_prob)

            # print('round2:y_min:{}, y_max:{}, entropy:{}'.format(y_min, y_max, entropy))

            if entropy > entropy_threshold:
                img_patches.append(img_org[y_min:y_max, :])

    return img_patches


def horizontal_lines_filt(lines_p, img_canny, debug):
    """
    file lines
    :param lines_p:
    :param img_canny:
    :param debug: if True, print debug info.
    :return:
    """
    height, width = img_canny.shape[0:2]
    width_threshold = width * 0.25

    # keep horizontal lines
    lines_horizontal = [line for line in lines_p[:, 0, :] if abs(line[1] - line[3]) <= 1]
    lines_horizontal_sorted = sorted(lines_horizontal, key=lambda x: x[1])
    lines_length = np.array([line[2] - line[0] for line in lines_horizontal_sorted])

    # fetch max lines
    fetch_seize = min(len(lines_length), 50)
    max_lines_index = lines_length.argsort()[-fetch_seize:][::-1]
    max_lines = [lines_length[index] for index in max_lines_index]
    minimal_in_max_line_length = min(max_lines)

    if debug:
        print('max length:', max_lines)

    # cluster lines into groups
    lines_group = []
    line_used_flag = [0 for line in lines_horizontal]  # record if line has been used
    index = 0
    while index < len(lines_horizontal_sorted):
        line = lines_horizontal_sorted[index]
        line_length = line[2] - line[0]
        if line_length < minimal_in_max_line_length:
            index += 1
            continue

        line_used_flag[index] = True
        max_y = line[1]
        if line[2] - line[0] > width_threshold:
            temp_group = []
            inner_index = 0
            while inner_index < len(lines_horizontal_sorted):
                inner_line = lines_horizontal_sorted[inner_index]
                if abs(inner_line[1] - max_y) <= 2 and abs(inner_line[1] - line[1]) <= 5:
                    temp_group.append(inner_line)
                    if (inner_line[2] - inner_line[0]) > line_length:
                        max_y = inner_line[1]
                    line_used_flag[inner_index] = True
                elif inner_index > index:  # && abs(inner_line[1] - line[1]) > 2
                    inner_index += len(lines_horizontal_sorted)

                inner_index += 1  # inner_index <= index

            lines_group.append(temp_group)

        index += 1

    # merge lines in group
    split_lines_info = []
    y_axis_list = [0]

    line_threshold = width * 0.7
    for group in lines_group:
        length, y_axis = merge_group(group, width, debug)
        if length > line_threshold:
            split_lines_info.append([0, y_axis, width, y_axis])
            y_axis_list.append(y_axis)
    y_axis_list.append(height)

    return split_lines_info, y_axis_list


def split_img_path(img_path, debug=False, plot=False):
    print('img_path', img_path)
    img = cv2.imread(img_path)
    split_img(img, debug, plot)


def split_img(img, debug=False, plot=False):
    if debug:
        print('image shape:', img.shape)
        start_time = time.time()

    img_org = img.copy()
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_canny = cv2.Canny(img_gray, 50, 120, L2gradient=False)
    img_canny = cv2.dilate(img_canny, None)

    try:
        lines_p = cv2.HoughLinesP(img_canny, 1, np.pi / 2, 100, minLineLength=10, maxLineGap=3)

        split_lines_info, y_axis_list = horizontal_lines_filt(lines_p, img_canny, debug)
        if len(split_lines_info) == 0:
            img_patches = []
        else:
            img_patches = cal_patches_entropy(img_gray, img_org, y_axis_list, debug)

        for x1, y1, x2, y2 in split_lines_info:  # for x1, y1, x2, y2 in lines_p[:, 0, :]:
            cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    except Exception as e:
        traceback.print_exc()
        return []

    if plot:
        end_time = time.time()
        print('spend time:', end_time - start_time)

        ratio = 0.5
        img_org_resize = cv2.resize(img_org, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_LINEAR)
        img_plot = cv2.resize(img, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_LINEAR)
        img_canny_plot = cv2.resize(img_canny, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_LINEAR)

        # cv2.imshow('org', img_org_resize)
        cv2.imshow('img_resize', img_plot)
        cv2.moveWindow('img_resize', 300, 0)
        cv2.imshow('img_canny', img_canny_plot)
        cv2.moveWindow('img_canny', 600, 0)

        patch_index = 1
        print('length of img_patches:', len(img_patches))
        for img_patch in img_patches:
            print('shape of img pathes', img_patch.shape)
            cv2.imshow('img_patches{}'.format(patch_index), img_patch)
            patch_index += 1

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return img_patches


if __name__ == '__main__':
    # for i in range(1, 11):
    #     split_img_path('data/long_img_{}.jpg'.format(i), debug=True, plot=True)
    # split_img_path('data/long_img_2.jpg', debug=True, plot=True)

    root_dir = 'split_data'
    for file in os.listdir(root_dir):
        if file.endswith('.jpg') or file.endswith('.png'):
            print(file)
            split_img_path(os.path.join(root_dir, file), debug=True, plot=True)

            # split_img_path(os.path.join(root_dir, 'TB2Ajd4bpkoBKNjSZFEXXbrEVXa_!!179242340.jpg'), debug=True, plot=True)
            # split_img_path(os.path.join(root_dir, 'T2z1BPXhXbXXXXXXXX_!!490358687.jpg'), debug=True, plot=True)
            # split_img_path(os.path.join(root_dir, 'TB2._gskVuWBuNjSszbXXcS7FXa_!!848073489.jpg'), debug=True, plot=True)
            # split_img_path(os.path.join(root_dir, 'TB1E0qEXVHM8KJjSZFwXXcibXXa_q90.jpg'), debug=True, plot=True)
            # split_img_path(os.path.join(root_dir, 'TB1TvSLaMLD8KJjSszeXXaGRpXa_q90.jpg'), debug=True, plot=True)
            # split_img_path(os.path.join(root_dir, 'TB29GF5aFOWBuNjy0FiXXXFxVXa_!!2893880789.jpg'), debug=True, plot=True)
