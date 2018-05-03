"""
Created by Alex Wang on 2018-03-30
test so file call
"""
import ctypes


def call_so():
    """
    test so file calling
    :return:
    """
    so = ctypes.CDLL('./sum.so')

    print("so.sum(50) = %d" % so.sum(50))
    so.display("hello world!")
    print("so.add() = %d" % so.add(ctypes.c_float(2), ctypes.c_float(2010)))


if __name__ == '__main__':
    call_so()
