"""
Created by Alex Wang
on 2018-05-14
"""

import os
import traceback
import time
import math

import numpy as np
import cv2

save_path = '/Users/alexwang/data/image_resize/result'


def not_white_graph(img, debug=False):
    """
    :param img:
    :param debug:
    :return:
    """
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    height, width = img.shape[0:2]

    check_height = int(height * 0.05)
    top_part = img_gray[0:check_height, :]
    bottom_part = img_gray[height - check_height: height, :]

    top_white_ratio = np.sum(top_part >= 230) / (check_height * width + math.exp(-6))
    bottom_white_ratio = np.sum(bottom_part >= 230) / (check_height * width + math.exp(-6))

    if top_white_ratio <= 0.5 or bottom_white_ratio <= 0.5:
        return True
    return False


def split_and_resize_wrap(img, debug=False, plot=False):
    """
    :param img:
    :param debug:
    :param plot:
    :return:
    """
    img_result = None
    if img.shape[2] > 3:  # 4 channel
        height, width = img.shape[0:2]
        img_bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        b, g, r, a = cv2.split(img)
        merged_alpha = cv2.merge([a, a, a])

        white_image = np.ones((height, width, 3), dtype=np.uint8) * 255  # white image
        img_org_blend = cv2.bitwise_and(img_bgr, merged_alpha)
        white_img_blend = cv2.bitwise_and(white_image,
                                          np.ones((height, width, 3), dtype=np.uint8) * 255 - merged_alpha)
        img_new = cv2.add(img_org_blend, white_img_blend)
        # if not_white_graph(img_new, debug):
        #     return None

        img_result = split_and_resize(img_new, debug=True, plot=plot)
    else:
        # if not_white_graph(img, debug):
        #     return None
        img_result = split_and_resize(img, debug=True, plot=plot)

    return img_result


def image_resize_fn(img_org, debug=False):
    """
    :param img_org:
    :param debug:
    :return:
    """
    target_height = 800
    target_width = 800
    target_tv_height = 434.
    try:
        img = img_org.copy()
        height, width = img.shape[0:2]
        ratio = target_tv_height / height
        img_resize = cv2.resize(img, (int(ratio * width), int(ratio * height)), interpolation=cv2.INTER_CUBIC)
        new_height, new_width = img_resize.shape[0:2]  # new_height == 440

        white_image = np.ones((target_height, target_width, 3), dtype=np.uint8) * 255

        x_min = int((target_width - new_width) / 2.)
        x_max = x_min + new_width
        y_min = 225
        y_max = y_min + new_height
        white_image[y_min:y_max, x_min:x_max, :] = img_resize

        if debug:
            print('[image_resize_fn] old_height:{}, old_width:{}'.format(height, width))
            print('[image_resize_fn] new_height:{}, new_width:{}'.format(new_height, new_width))
            print('[image_resize_fn] shape of image_resize result:{}'.format(white_image.shape))

        return white_image
    except Exception as e:
        traceback.print_exc()
        return None


def filt_contour(tv_contour_list, img_height, debug=False):
    """
    :param tv_contour_list:
    :param img_height:
    :param debug:
    :return:
    """
    center_y = img_height / 2.
    new_contour_list = []
    axis_list = []
    for contour in tv_contour_list:
        min_y = min([point[0][1] for point in contour])
        max_y = max([point[0][1] for point in contour])
        axis_list.append((min_y, max_y))

    abandon = [False for i in range(len(tv_contour_list))]
    for i in range(len(tv_contour_list)):
        for j in range(len(tv_contour_list)):
            min_y_one, max_y_one = axis_list[i]
            min_y_two, max_y_two = axis_list[j]
            length_one = max_y_one - min_y_one
            length_two = max_y_two - min_y_two
            y_overlap = (length_one + length_two -
                         (max(max_y_one, max_y_two) - min(min_y_one, min_y_two))) \
                        * 1.0 / min(length_one, length_two)

            if y_overlap <= 0.3:  # abandon one
                idx = i
                bool_one_contain = min_y_one <= center_y <= max_y_one
                bool_two_contain = min_y_two <= center_y <= max_y_two
                center_one = (max_y_one + min_y_one) / 2.
                center_two = (max_y_two + min_y_two) / 2.
                if bool_one_contain and not bool_two_contain:
                    idx = j
                elif bool_two_contain and not bool_one_contain:
                    idx = i
                elif abs(center_one - center_y) < abs(center_two - center_y):
                    idx = j
                else:
                    idx = i
                abandon[idx] = True
                if debug:
                    x, y = axis_list[idx]
                    print('[filt_contour] filt min_y:{} max_y:{}'.format(x, y))

    for i in range(len(tv_contour_list)):
        if not abandon[i]:
            new_contour_list.append(tv_contour_list[i])
    return new_contour_list


def split_and_resize(img, debug=False, plot=False):
    """
    get mainpart of image and resize to appropriate size and generate a new image of size 800*800
    :param img:
    :param debug:
    :param plot:
    :return:
    """
    final_result = None
    if debug:
        print('shape of image:', img.shape)

    start_time = time.time()
    img_org = img.copy()
    height, width = img.shape[0:2]
    mask = np.zeros([height + 2, width + 2, 1], np.uint8)
    # opencv blood fill to select background
    cv2.floodFill(img, mask, (height - 20, width - 20), (255, 255, 255), cv2.FLOODFILL_MASK_ONLY)
    cv2.floodFill(img, mask, (20, width - 20), (255, 255, 255), cv2.FLOODFILL_MASK_ONLY)
    cv2.floodFill(img, mask, (20, 20), (255, 255, 255), cv2.FLOODFILL_MASK_ONLY)
    cv2.floodFill(img, mask, (height - 20, 20), (255, 255, 255), cv2.FLOODFILL_MASK_ONLY)
    cv2.floodFill(img, mask, (height - 20, int(width / 2)), (255, 255, 255), cv2.FLOODFILL_MASK_ONLY)
    cv2.floodFill(img, mask, (20, int(width / 2)), (255, 255, 255), cv2.FLOODFILL_MASK_ONLY)

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    contour_mask = np.zeros((height, width, 3), dtype=np.uint8)
    white_image = np.ones((height, width, 3), dtype=np.uint8) * 255  # white image
    ret, img_mask = cv2.threshold(img_gray, 240, 255, cv2.THRESH_BINARY)
    img_mask_new = np.ones((height, width + 20), dtype=np.uint8) * 255
    img_mask_new[:, 10:width + 10] = img_mask
    img_mask_new = cv2.bitwise_not(img_mask_new)
    img_mask_new = cv2.dilate(img_mask_new, None)
    img_temp, contours, hierarchy = cv2.findContours(img_mask_new, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    tv_contour_list = []
    for contour in contours:
        min_x = min([point[0][0] for point in contour])
        max_x = max([point[0][0] for point in contour])
        min_y = min([point[0][1] for point in contour])
        max_y = max([point[0][1] for point in contour])
        # print('contour width:{}, height:{}'.format(max_x - min_x, max_y - min_y))

        if (max_x - min_x) > width * 0.5 \
                and (max_y - min_y) < height * 0.96 \
                and (max_y - min_y) > height * 0.3:
            print('tv_contour width:{}, height:{}'.format(max_x - min_x, max_y - min_y))
            contour = np.array([[[point[0][0] - 10, point[0][1]]] for point in contour])
            tv_contour_list.append(contour)
            cv2.fillPoly(contour_mask, pts=[contour], color=(255, 255, 255))

    end_time = time.time()
    if debug:
        print('elapse time:', end_time - start_time)
        print('contours:', len(contours))
        print(contours[0].shape)  # [k, 1, 2]
        print('tv_contours:', len(tv_contour_list))

    if len(tv_contour_list) >= 1:
        if len(tv_contour_list) >= 2:
            tv_contour_list = filt_contour(tv_contour_list, height, debug)
        tv_min_x = width
        tv_max_x = 0
        tv_min_y = height
        tv_max_y = 0
        for contour in tv_contour_list:
            tv_min_x = min(tv_min_x, min([point[0][0] for point in contour]))
            tv_max_x = max(tv_max_x, max([point[0][0] for point in contour]))
            tv_min_y = min(tv_min_y, min([point[0][1] for point in contour]))
            tv_max_y = max(tv_max_y, max([point[0][1] for point in contour]))

        # avoid exceed boundary
        tv_min_x = max(0, tv_min_x)
        tv_min_y = max(0, tv_min_y)
        tv_max_x = min(width, tv_max_x)
        tv_max_y = min(height, tv_max_y)

        # img_seg = img_org[tv_min_y:tv_max_y, tv_min_x:tv_max_x, :].copy()
        print('contour width:{}, height:{}'.format(tv_max_x - tv_min_x, tv_max_y - tv_min_y))

        img_org_blend = cv2.bitwise_and(img_org, contour_mask)
        white_img_blend = cv2.bitwise_and(white_image,
                                          np.ones((height, width, 3), dtype=np.uint8) * 255 - contour_mask)
        img_result = cv2.add(img_org_blend, white_img_blend)
        final_result = image_resize_fn(img_result[tv_min_y:tv_max_y, tv_min_x:tv_max_x, :], debug)

    if plot:
        # cv2.drawContours(img, contours, -1, (0, 0, 255), 3)
        contour_mask = cv2.dilate(contour_mask, None)
        img_in_contour = cv2.bitwise_and(img_org, contour_mask)
        cv2.drawContours(img, tv_contour_list, -1, (0, 255, 0), 3)
        cv2.imshow('img_org', img_org)
        cv2.imshow('img_mask', img_mask_new)
        cv2.imshow('img', img)
        cv2.imshow('contour mask', img_in_contour)
        if len(tv_contour_list) >= 1:
            cv2.imshow('img_result', final_result)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return final_result


def process_batch(dir_path):
    """
    :param dir_path:
    :return:
    """
    import shutil
    if os.path.exists(save_path):
        shutil.rmtree(save_path)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    file_name_list = [
        'TB2w_BnXNGYBuNjy0FnXXX5lpXa_!!3649421206.jpg',
        'TB1eRQsmvNNTKJjSspkwu1eWFXa.png',
        'TB1ix3AXfDH8KJjy1Xcwu3pdXXa.png',
        'TB28dHToH0kpuFjy0FjXXcBbVXa_!!2453680503.jpg',
        'TB2GO2vjSJjpuFjy0FdXXXmoFXa_!!263671308.jpg',
        'TB2qluDtFXXXXXKXXXXXXXXXXXX_!!2097705387.jpg',
        'TB2uD9Ydm3PL1JjSZFxXXcBBVXa_!!2555391676.jpg',
        'TB29XPrpv9TBuNjy1zbXXXpepXa_!!3221418665.jpg',
        'TB16UaiXvv85uJjSZFxwu1l4XXa.png',
        'TB1boN.RpXXXXb3XXXXwu0bFXXX.png',
        'TB1gj20ih6I8KJjSszfwu1ZVXXa.png',
        'TB2oTY2gYwTMeJjSszfXXXbtFXa_!!2780005954.jpg',
        'TB2xGKsoeySBuNjy1zdXXXPxFXa_!!2097705387.jpg',
        'TB1gaZ2bxPI8KJjSspfwu3CFXXa.png',
        'TB20h5Ifoo09KJjSZFDXXb9npXa_!!1972027918.png',
        'TB20LH0b22H8KJjy0FcXXaDlFXa_!!2072344000.png',
        'TB29NaYpM0kpuFjSspdXXX4YXXa_!!489673344.png',
        'TB2rc_OiiCYBuNkHFCcXXcHtVXa_!!751075004.png',
        'TB1k0bHkDTI8KJjSsphSuwFppXa.jpg'
    ]

    for file_name in file_name_list:
        # for file_name in os.listdir(dir_path):
        file_path = os.path.join(dir_path, file_name)
        print(file_path)
        img_result = None
        # file_path = os.path.join(dir_path, 'LB1N.wsXd3XBuNjt_n_XXcDSpXa.png_539.png')
        try:
            if os.path.isfile(file_path):
                img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
                result = split_and_resize_wrap(img, debug=True, plot=True)

                # save images
                if result is None:
                    print('result is None')
                else:
                    save_file = os.path.join(save_path, file_name)
                    cv2.imwrite(save_file, result)

        except Exception as e:
            traceback.print_exc()


def test_rgba(dir_path):
    file_name_list = [
        'TB29NaYpM0kpuFjSspdXXX4YXXa_!!489673344.png'
    ]
    for file_name in file_name_list:
        file_path = os.path.join(dir_path, file_name)
        print(file_path)
        # file_path = os.path.join(dir_path, 'LB1N.wsXd3XBuNjt_n_XXcDSpXa.png_539.png')
        try:
            if os.path.isfile(file_path):
                img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
                img_bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                b, g, r, a = cv2.split(img)
                print(np.max(a))
                merged = cv2.merge([a, a, a])
                img_new = cv2.bitwise_and(img_bgr, merged)
                cv2.imshow('img', img)
                cv2.imshow('b', b)
                cv2.imshow('g', g)
                cv2.imshow('r', r)
                cv2.imshow('a', a)
                cv2.imshow('img_new', img_new)

                cv2.waitKey(0)
                cv2.destroyAllWindows()
        except Exception as e:
            traceback.print_exc()


if __name__ == '__main__':
    print('test...')
    process_batch('./data')
    # test_rgba('/Users/alexwang/data/image_resize/image_org')
