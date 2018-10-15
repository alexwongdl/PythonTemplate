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


def test_broadcast():
    """
    不同形状的矩阵或数组加减乘除，如果形状不同，其中一个必须为1
    :return:
    """
    a_np = np.random.random(size=(32, 50, 128, 1))
    b_np = np.random.random(size=(1, 1, 128, 1024))
    c_np = a_np + b_np
    print('test_tf_plus, shape of c_np:{}'.format(c_np.shape))  # (32, 50, 128, 1024)


if __name__ == '__main__':
    test_numpy_array()
    test_broadcast()