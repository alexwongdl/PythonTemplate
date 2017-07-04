"""
Created by Alex Wang on 20170704
"""
import tensorflow as tf
import sys

def _read_words(filename):
    """
    :param filename:
    :return: 所有句子连成一个字符串，切分单词构成一个列表
    """
    with open(filename, "r") as f:
            return f.read().replace("\n", "<eos>").split()

def _build_vocab(filename):
    """
    :param filename:
    :return:
    """
    str_list = _read_words(filename)
    

if __name__ == "__main__":
    # str_list =_read_words("E://data/ptb/data/ptb.test.txt")
    # for str in str_list:
    #     print(str)

