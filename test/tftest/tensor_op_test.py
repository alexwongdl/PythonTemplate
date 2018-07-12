# -*- coding:utf8 -*-
"""
Created by Alex Wang
On 2018-07-11
"""
import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim


def test_unpool():
    """
    reverse op of pooling,
    convert a (B, H, W, C) tensor to (B, 2H, 2W, C)
    :return:
    """
    x_value = np.reshape(range(48), newshape=(2, 3, 4, 2))
    x = tf.constant(x_value)
    y = tf.tile(x, [1, 2, 2, 1])

    with tf.Session() as sess:
        y_value = sess.run(y)

    print(x_value)
    print(y_value)
    print(y_value.shape)

    print(x_value[0, 0, 0, 0])
    print(y_value[0, 0, 0, 0])
    print(y_value[0, 1, 0, 0])
    print(y_value[0, 1, 1, 0])
    print(y_value[0, 0, 1, 0])

    idx = np.argwhere(y_value <= 0)
    print(idx)


def test_conv2d_transpose():
    """
    解卷积
    :return: convert a (B, H, W, C1) tensor to (B, 2H, 2W, C2)
    """
    x = tf.random_uniform(shape=(32, 16, 16, 10))
    y = slim.conv2d_transpose(inputs=x, num_outputs=20, kernel_size=[3, 3], stride=2)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        y_value = sess.run(y)
    print(y_value.shape)


if __name__ == '__main__':
    test_unpool()
    test_conv2d_transpose()
