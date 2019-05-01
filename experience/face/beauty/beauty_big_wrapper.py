"""
Created by Alex Wang
On 2018-7-6
"""

"""
Created by Alex Wang
On 2018-06-27


convert 0~5 value to 0~50 score , use classification model
"""

import os
import sys
import time
from collections import namedtuple
import traceback

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import inception_v4
import inception_preprocessing

import cv2

model_path = os.path.join('beauty_model', 'model')  # big model


class FaceBeauty(object):
    def __init__(self, weight_file):
        self.image_size = inception_v4.inception_v4.default_image_size
        config = tf.ConfigProto(log_device_placement=False)
        config.gpu_options.allow_growth = True

        self.__graph = tf.Graph()

        with self.__graph.as_default():
            self.__session = tf.Session(config=config, graph=self.__graph)

            self.images_placeholder = tf.placeholder(tf.float32, shape=(None, None, 3),
                                                     name='image')
            img_process = inception_preprocessing.preprocess_for_eval(
                self.images_placeholder, self.image_size, self.image_size,
                central_fraction=1, scope='preprocess_test')
            img_process = tf.expand_dims(img_process, 0)

            arg_scope = inception_v4.inception_v4_arg_scope()
            with tf.variable_scope('big', reuse=False) as scope:
                with slim.arg_scope(arg_scope):
                    logits, end_points = inception_v4.inception_v4(img_process,
                                                                   is_training=False,
                                                                   num_classes=1001,
                                                                   dropout_keep_prob=1.0,
                                                                   reuse=False,
                                                                   create_aux_logits=False)

                with tf.variable_scope('Beauty', 'BeautyV1', reuse=False) as scope:
                    mid_full_conn = slim.fully_connected(end_points['PreLogitsFlatten'],
                                                         1000, activation_fn=tf.nn.relu,
                                                         scope='mid_full_conn',
                                                         trainable=False,
                                                         reuse=False)

                    predict_conn = slim.fully_connected(mid_full_conn,
                                                        100, activation_fn=None,
                                                        scope='100_class_conn',
                                                        trainable=False,
                                                        reuse=False)

                    beauty_weight = tf.convert_to_tensor([[i] for i in range(0, 100)], dtype=tf.float32)
                    regress_conn = tf.matmul(tf.nn.softmax(predict_conn), beauty_weight)  # 32 * 1

                    self.end_points = {'mid_full_conn': mid_full_conn,
                                       'regress_conn': regress_conn,
                                       'predict_conn': predict_conn}

            saver = tf.train.Saver(tf.global_variables(), max_to_keep=3)
            saver.restore(self.__session, weight_file)

    def infer(self, image):
        """
        :param image: shape=(None, None, 3)
        :return:
        """
        feed_dict = {self.images_placeholder: image}
        end_points_value = self.__session.run(self.end_points, feed_dict=feed_dict)
        return end_points_value


def test_one(img_path_list):
    batch_size = 1
    FLAG_tuple = namedtuple(typename='flag', field_names=['dropout', 'batch_size'])
    FLAGS = FLAG_tuple(dropout=1.0, batch_size=batch_size)

    FaceBeauty_instance = FaceBeauty(model_path)


    for img_path in img_path_list:
        start_time = time.time()
        try:
            img_org = cv2.imread(img_path)

            image_rgb = cv2.cvtColor(img_org, cv2.COLOR_BGR2RGB)
            image_rgb = image_rgb.astype(np.float32) / 255.
            result = FaceBeauty_instance.infer(image_rgb)
            beauty_score = round(result['regress_conn'][0][0], 3) / 20.
            print('image:{}, label:{}, result:{}'.format(img_path,
                                                         0,
                                                         beauty_score))
            print('mid level feature:{}'.format(result['mid_full_conn'][0]))
        except Exception as e:
            traceback.print_exc()
        end_time = time.time()
        print("cost time:{}".format(end_time - start_time))


if __name__ == '__main__':
    # test_model()

    img_path_list = [
        "/Users/alexwang/data/temp/5CC56E89.jpg",
        "/Users/alexwang/data/temp/D42EA9D6.jpg",
        "/Users/alexwang/data/temp/IMG_6645.JPG",
        "/Users/alexwang/data/temp/IMG_7376.JPG",
        "/Users/alexwang/data/temp/DingTalk20180706143012.png",
        "/Users/alexwang/data/temp/DingTalk20180706144656.png",
        "/Users/alexwang/data/temp/DingTalk20180706144709.png",
        "/Users/alexwang/data/temp/DingTalk20180706144721.png",
        "/Users/alexwang/data/temp/DingTalk20180706144731.png",
        "/Users/alexwang/data/temp/DingTalk20180706160032.png"
    ]
    test_one(img_path_list)
