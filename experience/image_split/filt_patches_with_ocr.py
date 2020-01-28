#! /usr/bin/python
# -*- coding: utf-8 -*-

"""
Created by Alex Wang
on 2018-04-28
"""

import traceback


def is_chinese(char):
    """
    judge is this char is a Chinese character
    :param char:
    :return: Boolean
    """
    if not char:
        return False

    if u'\u4e00' <= char <= u'\u9fa5':
        return True

    return False

def is_english(char):
    """
    judge is this char is a English character
    :param char:
    :return:
    """
    if not char:
        return False

    if u'\u0061' <= char <= u'\u007a':
        return True

    return False

def patches_filter_using_ocr(img_patches_axis_list, ocr_info, img_width, debug=False):
    """
    :param img_patches_axis_list: (y_min, y_max) tuple list
    :param ocr_info:
    :param img_width:
    :return:
    """
    ocr_info_list = []
    patches_chinese_count = [0 for patches in img_patches_axis_list]  # total chinese string length
    patches_max_chinese_len = [0 for patches in img_patches_axis_list]  # max chinese string length

    if not ocr_info or len(ocr_info) <= 1:
        return patches_chinese_count, patches_max_chinese_len

    elems = ocr_info.split(';')
    for elem in elems:
        try:
            sub_strs = elem.split(':')
            axises_str = sub_strs[0].split(',')
            axises = [int(axis) for axis in axises_str]
            min_x = min(axises[0::2])
            max_x = max(axises[0::2])
            min_y = min(axises[1::2])
            max_y = max(axises[1::2])

            words_str = sub_strs[1].split(',')[1]
            unicode_words_str = words_str.decode('utf-8')
            chinese_bool = [1 for char in unicode_words_str if is_chinese(char)]
            chinese_count = sum(chinese_bool)

            english_bool = [1 for char in unicode_words_str if is_english(char)]
            english_count = sum(english_bool)
            chinese_count += int(english_count / 4.0)

            ocr_info_list.append({'min_y': min_y, 'max_y': max_y,
                                  'min_x': min_x, 'max_x': max_x,
                                  'chinese_count': chinese_count})

            if debug:
                print('axises:{}'.format(axises))
                print('min_x:{}, max_x:{}, min_y:{}, max_y:{}'.format(min_x, max_x, min_y, max_y))
                print('words:{}'.format(words_str))
                print('chinese char count:{}'.format(chinese_count))
                print('')

        except Exception as e:
            traceback.print_exc()

    for i in range(len(img_patches_axis_list)):
        y_min, y_max = img_patches_axis_list[i]
        for ocr_dict in ocr_info_list:
            ocr_y_min = ocr_dict['min_y']
            ocr_y_max = ocr_dict['max_y']
            if max(y_min, ocr_y_min) < min(y_max, ocr_y_max):  # cross overlap
                patches_chinese_count[i] += ocr_dict['chinese_count']
                patches_max_chinese_len[i] = max(patches_max_chinese_len[i], ocr_dict['chinese_count'])

                # left-up corner
                if ocr_y_min <= (y_min + y_max) / 2 and ocr_dict['min_x'] <= img_width / 2:
                    patches_chinese_count[i] = 10000
                    patches_max_chinese_len[i] = 10000

    return patches_chinese_count, patches_max_chinese_len


def test_chiese():
    str = '252,59,537,59,537,72,252,72:99,【模特图仅供上身效果参考，宝贝以实物为准dsddc】;'
    # str = '模特图仅供上身效果参考，宝贝以实物为准】;'
    str_unicode = str.decode('utf-8')
    print(type(str))
    print(type(str_unicode))
    for char in str_unicode:
        char_utf8 = char.encode('utf-8')
        print("{}:{}, {}".format(char_utf8, is_chinese(char), is_english((char))))


def test_ocr_info():
    ocr_info = '318,25,472,25,472,42,318,42:99,人/模特展示;252,59,537,59,537,72,252,72:99,【模特图仅供上身效果参考，宝贝以实物为准dsdc】'
    patches_filter_using_ocr([], ocr_info, debug=True)


if __name__ == '__main__':
    test_chiese()
    # test_ocr_info()
