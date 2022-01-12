"""
 @author: AlexWang
 @date: 2022/1/13 12:01 上午
"""
import numpy as np
import io


def compress_numpy_to_bytes(numpy_arr):
    """
    numpy数组压缩成bytes
    :param numpy_arr:
    :return:
    """
    output = io.BytesIO()
    np.savez_compressed(output, numpy_arr)

    return list(output.getvalue())


def decompress_bytes_to_numpy(bytes_str):
    """
    bytes解压缩回numpy数组
    :param bytes_str:
    :return:
    """
    buffer = io.BytesIO(bytearray(bytes_str))
    arr = np.load(buffer)['arr_0']

    return arr


if __name__ == '__main__':
    # 测试例子，见download_util
    numpy_array = (255 * np.random.rand(256, 256)).astype(np.uint8)
    arr_bytes = compress_numpy_to_bytes(numpy_array)
    print("arr_bytes", type(arr_bytes))  # <class 'list'>
    print(arr_bytes)

    arr = decompress_bytes_to_numpy(arr_bytes)
    print(numpy_array)
    print(arr)

    diff = numpy_array - arr
    print("diff", np.sum(np.where(diff != 0, 1, 0)))