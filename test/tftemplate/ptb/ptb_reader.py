"""
Created by Alex Wang on 20170704
"""
import tensorflow as tf
import sys
import collections
import os

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
        word_to_id--单词到id的映射
        id_to_word--id到单词的映射
    """
    str_list = _read_words(filename)
    counter_one = collections.Counter(str_list)
    count_pairs =  sorted(counter_one.items(), key=lambda x:-x[1])
    words, _ = list(zip(*count_pairs))
    word_to_id = dict(zip(words, range(len(words))))
    id_to_word = dict(zip(range(len(words)), words))
    return word_to_id, id_to_word

def _file_to_word_ids(filename, word_to_id):
    """
    :param filename:
    :param word_to_id:
    :return:
        word_ids：单词对应的id列表
    """
    str_list = _read_words(filename)
    word_ids = [word_to_id[word] for word in str_list if word in word_to_id]
    return word_ids

def ptb_raw_data(data_path=None):
    """
    :param data_path:
    :return:
    """
    train_path = os.path.join(data_path, "ptb.train.txt")
    test_path = os.path.join(data_path, "ptb.test.txt")
    valid_path = os.path.join(data_path, "ptb.valid.txt")

    word_to_id, id_to_word = _build_vocab(train_path)
    train_data = _file_to_word_ids(train_path, word_to_id)
    test_data = _file_to_word_ids(test_path, word_to_id)
    valid_data = _file_to_word_ids(valid_path, word_to_id)
    return train_data, test_data, valid_data, word_to_id, id_to_word

def ptb_data_queue(raw_data, batch_size, num_steps):
    """
    range_input_producer构造数据输入queu
    :param raw_data:
    :param batch_size:
    :param num_steps:
    :return:
    """
    raw_data = tf.convert_to_tensor(raw_data, name="raw_data", dtype=tf.int32)
    data_len = tf.size(raw_data)
    batch_len = data_len // batch_size ## 每一个batch包含的单词个数
    data = tf.reshape(raw_data[0: batch_size * batch_len], [batch_size, batch_len])
    step_size = (batch_len - 1) // num_steps
    tf.strided_slice()

def test_ptb_data_queue():
    data_path = "E://data/ptb/data"
    train_data, test_data, valid_data, word_to_id, id_to_word = ptb_raw_data(data_path)
    sess = tf.Session()
    x, y = ptb_data_queue(train_data, batch_size=2000, num_steps=10)
    for i in range(10):
        print("round :" + str(i))
        x_value, y_value = sess.run([x,y])
        x_words = [id_to_word[id] for id in x_value if id in id_to_word]
        y_words = [id_to_word[id] for id in y_value if id in id_to_word]
        print("x_words:" + str(x_words))
        print("y_words:" + str(y_words))
    sess.close()

if __name__ == "__main__":
    # str_list =_read_words("E://data/ptb/data/ptb.test.txt")
    # for str in str_list:
    #     print(str)

