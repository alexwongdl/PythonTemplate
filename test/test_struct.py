"""
Created by Alex Wang
On 2018-08-26
"""
import struct


def decode_feature(feature, n, fmt="f"):
    """
    e.g.:img_features = decode_feature(img_features, 2048)
    :param feature:
    :param n:
    :param fmt:
    :return:
    """
    result = struct.unpack(fmt * n, feature)
    return result


def test_struct():
    """
    reference: http://ju.outofmemory.cn/entry/316
    :return:
    """
    a = 'hello'
    b = 'world!'
    c = 2
    d = 45.123

    bytes = struct.pack('5s6sif', a, b, c, d)
    print('bytes:{}'.format(bytes))

    unpack_result = struct.unpack('5s6sif', bytes)
    print('unpack_result:{}'.format(unpack_result))


if __name__ == '__main__':
    test_struct()
