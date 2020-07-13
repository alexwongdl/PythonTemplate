"""
Created by Alex Wang
On 2018-08-15
"""
import sys
import os

import tensorflow as tf
import tensorflow.contrib.slim as slim

sys.path.insert(0, 'test_inception')
import sphere_network
import inception_v4

image_width = 112
image_height = 112
embedding_size = 1024


def test_print_layer_info():
    """
    print layer name, input tensor and output tensor
    :return:
    """
    images_placeholder = tf.placeholder(tf.float32, shape=(
        None, image_height, image_width, 3), name='image')
    phase_train_placeholder = tf.placeholder(tf.bool, name='phase_train')

    prelogits = sphere_network.infer(images_placeholder, embedding_size)

    prelogits = slim.batch_norm(prelogits,
                                is_training=phase_train_placeholder,
                                epsilon=1e-5,
                                scale=True,
                                scope='softmax_bn')

    embeddings = tf.identity(prelogits)
    operations = tf.get_default_graph().get_operations()
    for operation in operations:
        print("Operation:{}".format(operation.name))
        for k in operation.inputs:
            print("{} Input: {}  {}".format(operation.name, k.name, k.get_shape()))
        for k in operation.outputs:
            print("{} Output:{}".format(operation.name, k.name))
        print("\n")


def test_inception_layer_info():
    """
    print layer name, input tensor and output tensor
    print example:
        Operation:InceptionV4/Logits/Logits/MatMul
        InceptionV4/Logits/Logits/MatMul Input: InceptionV4/Logits/PreLogitsFlatten/flatten/Reshape:0  (?, 1536)
        InceptionV4/Logits/Logits/MatMul Input: InceptionV4/Logits/Logits/weights/read:0  (1536, 1001)
        InceptionV4/Logits/Logits/MatMul Output:InceptionV4/Logits/Logits/MatMul:0


        Operation:InceptionV4/Logits/Logits/BiasAdd
        InceptionV4/Logits/Logits/BiasAdd Input: InceptionV4/Logits/Logits/MatMul:0  (?, 1001)
        InceptionV4/Logits/Logits/BiasAdd Input: InceptionV4/Logits/Logits/biases/read:0  (1001,)
        InceptionV4/Logits/Logits/BiasAdd Output:InceptionV4/Logits/Logits/BiasAdd:0


        Operation:InceptionV4/Logits/Predictions
        InceptionV4/Logits/Predictions Input: InceptionV4/Logits/Logits/BiasAdd:0  (?, 1001)
        InceptionV4/Logits/Predictions Output:InceptionV4/Logits/Predictions:0
    :return:
    """
    image_size = inception_v4.inception_v4.default_image_size
    inputs = tf.placeholder(dtype=tf.float32, shape=(None, image_size, image_size, 3))

    arg_scope = inception_v4.inception_v4_arg_scope()
    with slim.arg_scope(arg_scope):
        logits, end_points = inception_v4.inception_v4(inputs, is_training=False,
                                                       num_classes=1001)

    operations = tf.get_default_graph().get_operations()
    for operation in operations:
        print("Operation:{}".format(operation.name))
        for k in operation.inputs:
            print("{} Input: {}  {}".format(operation.name, k.name, k.get_shape()))
        for k in operation.outputs:
            print("{} Output:{}".format(operation.name, k.name))
        print("\n")


def test_inception_variable_info():
    """
    print every variable and its shape in graph
    print example:
        InceptionV4/Mixed_7d/Branch_2/Conv2d_0d_1x3/BatchNorm/moving_variance:0 (256,)
        InceptionV4/Mixed_7d/Branch_2/Conv2d_0e_3x1/weights:0 (3, 1, 512, 256)
        InceptionV4/Mixed_7d/Branch_2/Conv2d_0e_3x1/BatchNorm/beta:0 (256,)
        InceptionV4/Mixed_7d/Branch_2/Conv2d_0e_3x1/BatchNorm/moving_mean:0 (256,)
        InceptionV4/Mixed_7d/Branch_2/Conv2d_0e_3x1/BatchNorm/moving_variance:0 (256,)
        InceptionV4/Mixed_7d/Branch_3/Conv2d_0b_1x1/weights:0 (1, 1, 1536, 256)
        InceptionV4/Mixed_7d/Branch_3/Conv2d_0b_1x1/BatchNorm/beta:0 (256,)
        InceptionV4/Mixed_7d/Branch_3/Conv2d_0b_1x1/BatchNorm/moving_mean:0 (256,)
        InceptionV4/Mixed_7d/Branch_3/Conv2d_0b_1x1/BatchNorm/moving_variance:0 (256,)
        InceptionV4/AuxLogits/Conv2d_1b_1x1/weights:0 (1, 1, 1024, 128)
        InceptionV4/AuxLogits/Conv2d_1b_1x1/BatchNorm/beta:0 (128,)
        InceptionV4/AuxLogits/Conv2d_1b_1x1/BatchNorm/moving_mean:0 (128,)
        InceptionV4/AuxLogits/Conv2d_1b_1x1/BatchNorm/moving_variance:0 (128,)
        InceptionV4/AuxLogits/Conv2d_2a/weights:0 (5, 5, 128, 768)
        InceptionV4/AuxLogits/Conv2d_2a/BatchNorm/beta:0 (768,)
        InceptionV4/AuxLogits/Conv2d_2a/BatchNorm/moving_mean:0 (768,)
        InceptionV4/AuxLogits/Conv2d_2a/BatchNorm/moving_variance:0 (768,)
        InceptionV4/AuxLogits/Aux_logits/weights:0 (768, 1001)
        InceptionV4/AuxLogits/Aux_logits/biases:0 (1001,)
        InceptionV4/Logits/Logits/weights:0 (1536, 1001)
        InceptionV4/Logits/Logits/biases:0 (1001,)
    :return:
    """
    image_size = inception_v4.inception_v4.default_image_size
    inputs = tf.placeholder(dtype=tf.float32, shape=(None, image_size, image_size, 3))

    arg_scope = inception_v4.inception_v4_arg_scope()
    with slim.arg_scope(arg_scope):
        logits, end_points = inception_v4.inception_v4(inputs, is_training=False,
                                                       num_classes=1001)
    for tensor in tf.all_variables():
        print(tensor.name, tensor.shape)


def test_checkpoint_variables():
    """
    print every variable and its shape in checkpoint
    print example:
        InceptionV4/AuxLogits/Aux_logits/biases [1001]
        InceptionV4/AuxLogits/Aux_logits/weights [768, 1001]
        InceptionV4/AuxLogits/Conv2d_1b_1x1/BatchNorm/beta [128]
        InceptionV4/AuxLogits/Conv2d_1b_1x1/BatchNorm/moving_mean [128]
        InceptionV4/AuxLogits/Conv2d_1b_1x1/BatchNorm/moving_variance [128]
        InceptionV4/AuxLogits/Conv2d_1b_1x1/weights [1, 1, 1024, 128]
        InceptionV4/AuxLogits/Conv2d_2a/BatchNorm/beta [768]
        InceptionV4/AuxLogits/Conv2d_2a/BatchNorm/moving_mean [768]
        InceptionV4/AuxLogits/Conv2d_2a/BatchNorm/moving_variance [768]
        InceptionV4/AuxLogits/Conv2d_2a/weights [5, 5, 128, 768]
    :return:
    """
    model_dir = '/Users/alexwang/workspace/vip_image_face_analysis/data/model'
    checkpoint_path = os.path.join(model_dir, 'model')
    model_dir = '/Users/alexwang/data/video_select/train_beauty_log'
    checkpoint_path = os.path.join(model_dir, 'save-28999-0.926119962159')
    for var_name, shape in tf.contrib.framework.list_variables(checkpoint_path):
        print(var_name, shape)

if __name__ == '__main__':
    # test_inception_layer_info()
    # test_inception_variable_info()
    test_checkpoint_variables()