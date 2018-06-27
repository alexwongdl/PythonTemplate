"""
Created by Alex Wang
on 2018-05-15
"""
import os

import tensorflow as tf
import cv2
import numpy as np
import tensorflow.contrib.slim as slim
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file

import inception_v4
from test.tftemplate.data import imagenet
from test.tftemplate.data import inception_preprocessing

label_names = imagenet.create_readable_names_for_imagenet_labels()
print(label_names)


def test():
    model_dir = '/Users/alexwang/data/'
    checkpoint_path = os.path.join(model_dir, 'inception_v4.ckpt')
    img_path = '../data/laska.png'
    img = cv2.imread(img_path)
    print('shape of img:', img.shape)

    # preprocess image
    batch_size = 2
    image_size = inception_v4.inception_v4.default_image_size
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.
    preprocessed_image_one = inception_preprocessing.preprocess_for_eval(
        tf.convert_to_tensor(img_rgb, tf.float32), image_size, image_size,
        central_fraction=1)

    zeros_mat = np.zeros(shape=(image_size, image_size, 3), dtype=np.float32)
    preprocessed_image_two = inception_preprocessing.preprocess_for_eval(
        tf.convert_to_tensor(zeros_mat, tf.float32), image_size, image_size,
        central_fraction=1)
    inputs = tf.concat([tf.expand_dims(preprocessed_image_one, 0),
                        tf.expand_dims(preprocessed_image_two, 0)], axis=0)

    # construct net
    # print_tensors_in_checkpoint_file(checkpoint_path, None, False)
    arg_scope = inception_v4.inception_v4_arg_scope()
    with slim.arg_scope(arg_scope):
        logits, end_points = inception_v4.inception_v4(inputs, is_training=False,
                                                       num_classes=1001)
    for tensor in tf.all_variables():
        print(tensor.name, tensor.shape)

    probabilities = tf.nn.softmax(logits)

    # restore net
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, checkpoint_path)

        # predict
        inputs_value, probabilitie_values, end_points_values = sess.run(
            [inputs, probabilities, end_points])
        print('shape of inputs:', inputs_value.shape)
        print('shape of predict', probabilitie_values.shape)
        # print('end_point:', end_points_values)
        for key in end_points_values.keys():
            print(key, end_points_values[key].shape)

        for predict in probabilitie_values:
            sorted_predict = np.argsort(predict)
            top_5 = sorted_predict[::-1][0:5]
            top_5_labels = [(label_names[i], predict[i]) for i in top_5]
            print(top_5_labels)


if __name__ == '__main__':
    test()
