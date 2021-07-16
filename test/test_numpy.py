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
    ones_arr = np.ones((5, 5), dtype=np.uint8)
    part = ones_arr[2:4, 2:4]
    print(ones_arr)
    part[0, 0] = 0
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


def test_npz():
    """
    numpy矩阵可以保存在npy或者npz文件
    :return:
    """
    str_a = 'abc'
    dict_a = {'name': 'alexwang', 'age': 20}
    arr_a = np.array(range(12)).reshape((3, 4))

    np.savez('test.npz', str_a=str_a, dict_a=dict_a, arr_a=arr_a)

    data = np.load('test.npz')
    print('arr_a:{}'.format(data['arr_a']))
    print('str_a:{}'.format(data['str_a']))
    print('dict_a:{}'.format(data['dict_a']))


def test_tostring_fromstring():
    """

    :return:
    """
    arr = np.reshape(np.array(range(20)), newshape=(4, 5))
    print(arr)
    arr_str = arr.tostring()
    arr_rec = np.fromstring(arr_str, dtype=np.int32)
    print(arr_rec)


def test_chunk():
    """
    10 : (5, 256, 256, 3)
        (5, 256, 256, 3)
    9: (5, 256, 256, 3)
        (4, 256, 256, 3)
    4: (4, 256, 256, 3)
    2: (2, 256, 256, 3)
    6: (5, 256, 256, 3)
        (1, 256, 256, 3)
    :return:
    """
    arr = np.random.random((6, 256, 256, 3))

    chunk_size = 5
    chunks = [arr[i:i + chunk_size, :, :, :] for i in range(0, arr.shape[0], chunk_size)]
    for chunk in chunks:
        print(chunk.shape)


if __name__ == '__main__':
    # test_numpy_array()
    # test_broadcast()
    # test_npz()
    # test_tostring_fromstring()
    test_chunk()