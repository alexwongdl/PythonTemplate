"""
Created by Alex Wang
on 2018-05-14
"""
import os
import traceback
import time

import cv2
import numpy as np


def horizontal_edge_length_in_canny(img_canny, y_axis):
    """
    horizontal line detection in x_axis on img_canny
    :param img_canny:
    :param y_axis:
    :return:
    """
    white_pixel_num = 0
    line_pixels = img_canny[y_axis, :]
    for pixel in line_pixels:
        if pixel >= 200:
            white_pixel_num += 1
    return white_pixel_num


def vertical_edge_length_in_canny(img_canny, x_axis):
    """
    vertical line detection in x_axis on img_canny
    :param img_canny:
    :param x_axis:
    :return:
    """
    white_pixel_num = 0
    line_pixels = img_canny[:, x_axis]
    for pixel in line_pixels:
        if pixel >= 200:
            white_pixel_num += 1
    return white_pixel_num


def verticle_lines_filt(lines_p, img_canny, debug):
    """
    filter lines
    :param lines_p:
    :param img_canny:
    :param debug: if True, print debug info.
    :return:
    """
    height, width = img_canny.shape[0:2]
    height_threshold = height * 0.25
    line_threshold = height * 0.7

    # keep vertical lines
    lines_vertical = [line for line in lines_p[:, 0, :] if abs(line[0] - line[2]) <= 1]
    lines_vertical_sorted = sorted(lines_vertical, key=lambda x: x[0])
    lines_length = np.array([abs(line[3] - line[1]) for line in lines_vertical_sorted])
    if len(lines_length) == 0:
        return []

    # fetch max lines
    fetch_seize = min(len(lines_length), 20)
    max_lines_index = lines_length.argsort()[-fetch_seize:][::-1]
    max_lines = [lines_length[index] for index in max_lines_index]
    minimal_in_max_line_length = min(max_lines)

    if debug:
        print('max length:', max_lines)

    x_axis_list = []
    for line in lines_vertical_sorted:
        temp_line_length = abs(line[3] - line[1])
        if temp_line_length < height_threshold:
            continue
        x_axis = line[0]
        if vertical_edge_length_in_canny(img_canny, x_axis) > line_threshold:  # TODO:need validation
            x_axis_list.append(x_axis)

    return x_axis_list


def horizontal_lines_filt(lines_p, img_canny, debug):
    """
    filter lines
    :param lines_p:
    :param img_canny:
    :param debug: if True, print debug info.
    :return:
    """
    height, width = img_canny.shape[0:2]
    width_threshold = width * 0.25
    line_threshold = width * 0.7

    # keep horizontal lines
    lines_horizontal = [line for line in lines_p[:, 0, :] if abs(line[1] - line[3]) <= 1]
    lines_horizontal_sorted = sorted(lines_horizontal, key=lambda x: x[1])
    lines_length = np.array([abs(line[2] - line[0]) for line in lines_horizontal_sorted])
    if len(lines_length) == 0:
        return []

    # fetch max lines
    fetch_seize = min(len(lines_length), 20)
    max_lines_index = lines_length.argsort()[-fetch_seize:][::-1]
    max_lines = [lines_length[index] for index in max_lines_index]
    minimal_in_max_line_length = min(max_lines)

    if debug:
        print('max length:', max_lines)

    y_axis_list = []
    for line in lines_horizontal_sorted:
        temp_line_length = abs(line[2] - line[0])
        if temp_line_length < width_threshold:
            continue
        y_axis = line[1]
        if horizontal_edge_length_in_canny(img_canny, y_axis) > line_threshold:  # TODO:need validation
            y_axis_list.append(y_axis)

    return y_axis_list


def filt(img, debug=False, plot=False):
    start_time = time.time()
    img_org = img.copy()
    height, width = img_org.shape[0:2]

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, img_mask = cv2.threshold(img_gray, 250, 255, cv2.THRESH_BINARY)
    img_canny = cv2.Canny(img_mask, 50, 120, L2gradient=False)
    img_canny = cv2.dilate(img_canny, None)
    x_axis_list = []
    y_axis_list = []
    try:
        lines_p = cv2.HoughLinesP(img_canny, 1, np.pi / 2, 100, minLineLength=10, maxLineGap=3)
        if lines_p is not None:
            x_axis_list = verticle_lines_filt(lines_p, img_canny, debug)
            y_axis_list = horizontal_lines_filt(lines_p, img_canny, debug)
    except Exception as e:
        traceback.print_exc()

    be_filtered = False
    x_axis_min_bound = 0.1 * width
    x_axis_max_bound = 0.9 * width
    y_axis_min_bound = 0.1 * height
    y_axis_max_bound = 0.9 * height
    for x_axis in x_axis_list:
        if x_axis_min_bound <= x_axis <= x_axis_max_bound:
            be_filtered = True
            break
    for y_axis in y_axis_list:
        if y_axis_min_bound <= y_axis <= y_axis_max_bound:
            be_filtered = True
            break

    x_axis_list_org = []
    y_axis_list_org = []
    if not be_filtered:
        img_org_canny = cv2.Canny(img_gray, 50, 120, L2gradient=False)
        img_org_canny = cv2.dilate(img_org_canny, None)
        try:
            lines_p_org = cv2.HoughLinesP(img_org_canny, 1, np.pi / 2, 100, minLineLength=10, maxLineGap=3)
            if lines_p_org is not None:
                x_axis_list_org = verticle_lines_filt(lines_p_org, img_canny, debug)
                y_axis_list_org = horizontal_lines_filt(lines_p_org, img_canny, debug)
        except Exception as e:
            traceback.print_exc()
        for x_axis in x_axis_list_org:
            if x_axis_min_bound <= x_axis <= x_axis_max_bound:
                be_filtered = True
                break
        for y_axis in y_axis_list_org:
            if y_axis_min_bound <= y_axis <= y_axis_max_bound:
                be_filtered = True
                break


    circles = None
    if not be_filtered:
        circles = cv2.HoughCircles(img_canny, cv2.HOUGH_GRADIENT, 1.1, 100, param2=120)
        width4 = 0.4 * width
        height4 = 0.4 * height
        width45 = 0.45 * width
        width55 = 0.55 * width
        height35 = 0.35 * height
        height65 = 0.65 * height
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            if debug:
                print('run hough circle')
                print('circles:', circles)
            for (x, y, r) in circles:
                if r >= width4 or r >= height4:
                    if width45 <= x <= width55 and height35 <= y <= height65:
                        be_filtered = True

    end_time = time.time()
    if plot and be_filtered:
        print('be filtered:', be_filtered)
        print('elapse time:', end_time - start_time)
        print('image shape:', img_org.shape)
        print('x_axis_list:', x_axis_list)
        print('y_axis_list:', y_axis_list)
        for x_axis in x_axis_list:
            cv2.line(img, (x_axis, 0), (x_axis, height), (0, 0, 255), 2)
        for y_axis in y_axis_list:
            cv2.line(img, (0, y_axis), (width, y_axis), (0, 0, 255), 2)
        for x_axis in x_axis_list_org:
            cv2.line(img, (x_axis, 0), (x_axis, height), (0, 255, 0), 2)
        for y_axis in y_axis_list_org:
            cv2.line(img, (0, y_axis), (width, y_axis), (0, 255, 0), 2)
        if circles is not None:
            for (x, y, r) in circles:
                cv2.circle(img, (x, y), r, (0, 0, 255), 2)
        cv2.imshow('img_canny', img_canny)
        cv2.imshow('img_org', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return be_filtered


def process_batch(dir_path):
    """
    :param dir_path:
    :return:
    """
    img_name_list = [
        # 'TB2.Xw5oDnI8KJjSszgXXc8ApXa_!!832551907_341.jpg',
        # 'TB21NMpcCz9F1JjSZFMXXXmNXXa_!!2438930156_556.jpg',
        # 'TB2ab2scKuSBuNjSsplXXbe8pXa_!!671493136_11.jpg',
        # 'TB2cetdthdkpuFjy0FbXXaNnpXa_!!2454264124_402.jpg',
        # circle
        'TB2ylCtclU4h1JjSZFLXXaFMpXa_!!2454264124_401.jpg',
        'TB22RR1vJBopuFjSZPcXXc9EpXa_!!2181576422_384.jpg',
        'TB23VTijhtmpuFjSZFqXXbHFpXa_!!735040693_225.jpg',
        'TB26bLglHJmpuFjSZFwXXaE4VXa_!!2773481216_222.jpg',
        'TB2_QplhH1YBuNjSszhXXcUsFXa_!!732334263_369.jpg',
        'TB2_WZBramWBuNjy1XaXXXCbXXa_!!2356733680_972.jpg'
    ]
    for file_name in os.listdir(dir_path):
    # for file_name in img_name_list:
        file_path = os.path.join(dir_path, file_name)
        print(file_path)

        try:
            if os.path.isfile(file_path):
                img = cv2.imread(file_path)
                filt(img, debug=True, plot=True)

        except Exception as e:
            traceback.print_exc()


if __name__ == '__main__':
    process_batch('/Users/alexwang/data/image_split/image_split_filter_result')
