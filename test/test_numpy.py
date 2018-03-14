"""
Created by Alex Wang on 2018-03-14
"""

import numpy as np

def test_numpy_array():
    """
    获取部分数组只是引用，不是真的复制
    复制需要使用copy函数
    :return:
    """
    ones_arr = np.ones((5,5), dtype=np.uint8)
    part = ones_arr[2:4, 2:4]
    print(ones_arr)
    part[0,0] = 0
    print(ones_arr)


if __name__ == '__main__':
    test_numpy_array()