"""
Created by Alex Wang on 2018-02-03
"""

import cv2

def color_space():
    """
    打印所有可能的颜色空间转换
    :return:
    """
    flags = [i for i in dir(cv2) if i.startswith('COLOR_BGR')]
    [print(flag) for flag in flags]

if __name__ == '__main__':
    color_space()