"""
Created by Alex Wang
on 2018-05-11

Please refer to:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.md
for detail for pre-trained models download.
"""
import os

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
import cv2

import mobilenet_v1

print(tf.__version__)


def load_mobilenet_meta():
    model_dir = '/Users/alexwang/data/mobilenet_v1_1.0_160'
    meta_dir = os.path.join(model_dir, 'mobilenet_v1_1.0_160.ckpt.meta')

    mobilenet_saver = tf.train.import_meta_graph(meta_dir)
    mobilenet_graph = tf.get_default_graph()

    for tensor in tf.all_variables():
        print(tensor.name, tensor.shape)


def load_mobilenet_pb():
    model_dir = '/Users/alexwang/data/mobilenet_v1_1.0_160'
    pb_dir = os.path.join(model_dir, 'mobilenet_v1_1.0_160_frozen.pb')

    mobilenet_graph = tf.Graph()
    with mobilenet_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(pb_dir, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    with mobilenet_graph.as_default():
        for op in mobilenet_graph.get_operations():
            print(op.name)

            # input = tf.get_default_graph().get_tensor_by_name('input:0')


def test_mobilenet_v1():
    batch_size = 5
    height, width = 224, 224
    num_classes = 1001

    img = cv2.imread('laska.png')
    img_resize = cv2.resize(img, (height, width))
    img_rgb = cv2.cvtColor(img_resize, cv2.COLOR_BGR2RGB).astype(np.float32)
    # input_mat = np.random.rand(batch_size, height, width, 3).astype(np.float32)
    input_mat = np.zeros(shape=(batch_size, height, width, 3), dtype=np.float32)
    input_mat[0, :, :, :] = (img_rgb - 127) / 127.0
    print(input_mat[0, :, :, :])

    # inputs = tf.random_uniform((batch_size, height, width, 3), name='input')
    inputs = tf.convert_to_tensor(input_mat, name='input', dtype=tf.float32)

    arg_scope = mobilenet_v1.mobilenet_v1_arg_scope(is_training=False, weight_decay=0.0)
    with slim.arg_scope(arg_scope):
        logits, end_points = mobilenet_v1.mobilenet_v1(inputs, num_classes,
                                                   is_training=False)

    model_dir = '/Users/alexwang/data/mobilenet_v1_1.0_160'
    checkpoint_path = os.path.join(model_dir, 'mobilenet_v1_1.0_160.ckpt')
    print_tensors_in_checkpoint_file(checkpoint_path, None, False)
    saver = tf.train.Saver()

    # print all node in graph
    # for tensor in tf.get_default_graph().as_graph_def().node:
    #     print(tensor.name)

    input_get = tf.get_default_graph().get_tensor_by_name('input:0')
    print('shape of input_get:{}'.format(input_get.shape))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        predict = sess.run([end_points['Predictions']])
        print(predict)
        classes = np.argmax(predict[0], axis=1)
        print(classes)

        saver.restore(sess, checkpoint_path)
        predict = sess.run([end_points['Predictions']])
        print(predict)
        classes = np.argsort(predict[0], axis=1)
        for predict_result in classes:
            print(predict_result[::-1][0:5])


if __name__ == '__main__':
    # load_mobilenet_meta()
    # print('-----------------------------------------------------------')
    # load_mobilenet_pb()
    test_mobilenet_v1()
