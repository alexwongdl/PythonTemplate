"""
Created by Alex Wang
On 2018-07-19

test resnet
"""
import os
import tensorflow as tf
import tensorflow.contrib.slim as slim

import resnet_v1
import resnet_utils

image_size = resnet_v1.resnet_v1.default_image_size

def resnet_tensorboard():
    x_input = tf.placeholder(dtype=tf.float32, shape=(None, image_size, image_size, 3))
    arg_scope = resnet_utils.resnet_arg_scope()
    with slim.arg_scope(arg_scope):
        logits_50, end_points_50 = resnet_v1.resnet_v1_50(x_input,
                                                    num_classes=1000,
                                                    is_training=False,
                                                    global_pool=True,
                                                    output_stride=None,
                                                    spatial_squeeze=True,
                                                    store_non_strided_activations=False,
                                                    reuse=False,
                                                    scope='resnet_v1_50')

        logits_101, end_points_101 = resnet_v1.resnet_v1_101(x_input,
                                                          num_classes=1000,
                                                          is_training=False,
                                                          global_pool=True,
                                                          output_stride=None,
                                                          spatial_squeeze=True,
                                                          store_non_strided_activations=False,
                                                          reuse=False,
                                                          scope='resnet_v1_101')

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config= config) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        summary_writer = tf.summary.FileWriter('/Users/alexwang/data/resnet_summary', graph=sess.graph)
        summary_writer.close()

if __name__ == '__main__':
    resnet_tensorboard()