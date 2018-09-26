#!/usr/bin/python
# -*- coding:utf8 -*-
"""
Created by Alex Wang
On 2018-09-18
"""
import os
import json
import base64

import tensorflow as tf
import cv2
import numpy as np

def test_vlad():
    print('test vlad...')
    word_num = 2
    d = 2
    curr_batch_size = 1
    frame_num = 3

    nets = tf.Variable([[[10, 15], [20, 25], [30, 35]]], dtype=np.float32, name='input')
    centers = tf.Variable([[1, 1], [2, 2]], dtype=np.float32, name='centers')  # (word_num, d)

    with tf.variable_scope('distance') as scope:
        # (batch_size, frame_num * word_num, d)
        frame_tile = tf.tile(nets, multiples=[1, 1, word_num], name='frame_tile')
        frame_reshape = tf.reshape(frame_tile, shape=(curr_batch_size, frame_num * word_num, d),
                                   name='frame')
        #  (1, word_num, d)
        centers_expand = tf.expand_dims(centers, dim=0)

        # (batch_size, word_num, frame_num * d)
        centers_tile = tf.tile(centers_expand, multiples=[curr_batch_size, frame_num, 1],
                               name='centers_tile')

        # (batch_size, frame_num * word_num, d)
        substract = frame_reshape - centers_tile
        substract_reshape = tf.reshape(substract, shape=(curr_batch_size, frame_num, word_num, d))

    with tf.variable_scope('similar') as scope:
        # (batch_size, frame_num, word_num)
        distance = tf.Variable([[[10, 1], [20, 1], [100, 1]]], dtype=np.float32, name='distance')
        softmax = tf.nn.softmax(distance , dim=-1, name='dist_softmax')

        softmax_expand = tf.expand_dims(softmax, dim=-1)
        softmax_tile = tf.tile(softmax_expand, multiples=[1, 1, 1, d])

    result = tf.multiply(softmax_tile, substract_reshape)
    result_1 = tf.reduce_sum(result, axis=1)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        centers_tile_v = sess.run(centers_tile)
        print('centers_tile:{}'.format(centers_tile_v))

        frame_tile_v = sess.run(frame_reshape)
        print('frame_tile{}'.format(frame_tile_v))

        sub_v = sess.run(substract)
        print('substract:{}'.format(sub_v))

        substract_v = sess.run(substract_reshape)
        print('substract_reshape_v:{}'.format(substract_v))
        print(substract_v.shape)

        softmax_tile_v = sess.run(softmax_tile)
        print('softmax_tile_v:{}'.format(softmax_tile_v))

        result_v = sess.run(result)
        print('result:{}'.format(result_v))

        result_1_v = sess.run(result_1)
        print('result_1:{}'.format(result_1_v))

if __name__ == '__main__':
    test_vlad()
