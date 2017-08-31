"""
Create by Alex Wang
On 2017-08-31

tensorlayer 共享变量 https://github.com/wagamamaz/tensorlayer-tricks
"""
##inception 接FC 构建图片去重网路，inception在预训练模型上finetune
import numpy as np
import tensorflow as tf
import tensorlayer as tl
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.python.slim.nets.alexnet import alexnet_v2
from tensorflow.contrib.slim.python.slim.nets.inception_v3 import inception_v3_base, inception_v3, inception_v3_arg_scope
import skimage
import skimage.io
import skimage.transform
from scipy.misc import imread, imresize
import time, os
os.environ["CUDA_VISIBLE_DEVICES"]=""

def inception_layer(net_in, reuse=None):
    with slim.arg_scope(inception_v3_arg_scope()):
        ## Alternatively, you should implement inception_v3 without TensorLayer as follow.
        # logits, end_points = inception_v3(X, num_classes=1001,
        #                                   is_training=False)
        network = tl.layers.SlimNetsLayer(layer=net_in, slim_layer=inception_v3,
                                          slim_args= {
                                              'num_classes' : 1001,
                                              'is_training' : False,
                                              #  'dropout_keep_prob' : 0.8,       # for training
                                              #  'min_depth' : 16,
                                              #  'depth_multiplier' : 1.0,
                                              #  'prediction_fn' : slim.softmax,
                                              #  'spatial_squeeze' : True,
                                              'reuse' : reuse,
                                              #  'scope' : 'InceptionV3'
                                          },
                                          name='InceptionV3'  # <-- the name should be the same with the ckpt model
                                          )
    return network

## 构建网络
x_1 = tf.placeholder(tf.float32, shape=[None, 299, 299, 3], name='x-1')
x_2 = tf.placeholder(tf.float32, shape=[None, 299, 299, 3], name='x_2')
net_in_1 = tl.layers.InputLayer(x_1, name='input_layer_1')
net_in_2 = tl.layers.InputLayer(x_2, name='input_layer_2')

tl.layers.set_name_reuse(False)
inception_layer_1 = inception_layer(net_in_1, None)
tl.layers.set_name_reuse(True)
inception_layer_2 = inception_layer(net_in_2, True)

sess = tf.InteractiveSession()
inception_layer_1.print_params(False)
saver = tf.train.Saver()
saver.restore(sess, "/home/recsys/hzwangjian1/tensorflow/test/inception_v3.ckpt")

y_1 = inception_layer_1.outputs
probs_1 = tf.nn.softmax(y_1, name = 'probs_1')

y_2 = inception_layer_2.outputs
probs_2 = tf.nn.softmax(y_2, name = 'probs_2')

with tf.name_scope('total_net') as scope:
    inception_concat_layer = tl.layers.ConcatLayer([inception_layer_1, inception_layer_2], name='inception_concat_layer')
    concat_layer_result = inception_concat_layer.outputs
    fc_layer = tl.layers.DenseLayer(inception_concat_layer, n_units=10000, act=tf.nn.relu, name='fc_layer')
    fc_layer_two = tl.layers.DenseLayer(fc_layer, n_units=2, act=tf.nn.relu, name='fc_layer_two')
    result = tf.nn.softmax(fc_layer_two.outputs)



## inception-v3测试，class-names用https://newsrec92.dg.163.org:9999/notebooks/hzwangjian1/tensorflow/test/bottleneck_feed.ipynb
def load_image(path):
    # load image
    img = imread(path, mode='RGB')
    img = img / 255.0
    print(img.shape)
    assert (0 <= img).all() and (img <= 1.0).all()
    # print "Original Image Shape: ", img.shape
    # we crop image from center
    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy: yy + short_edge, xx: xx + short_edge]
    # resize to 299, 299
    resized_img = skimage.transform.resize(crop_img, (299, 299))
    return resized_img

result_summary_1 = tf.summary.tensor_summary(probs_1.op.name, probs_1)
result_summary_2 = tf.summary.tensor_summary(probs_2.op.name, probs_2)
tf_wirter = tf.summary.FileWriter('/data/hzwangjian1/tensorflow/tensorboard', sess.graph)
summary_op = tf.summary.merge_all()
sess.run(tf.global_variables_initializer())

img1 = load_image("/home/recsys/hzwangjian1/tensorflow/test/puzzle.jpeg") # test data in github: https://github.com/zsdonghao/tensorlayer/tree/master/example/data
# img1 = load_image("/data/hzwangjian1/tensorflow/laska.png")
img_list = [img1]
# img1 = img1.resize((1, 299, 299, 3))

import matplotlib.pyplot as plt
fig = plt.figure()
plt.imshow (img1)
plt.show()


start_time = time.time()
prob, concat_result, summary = sess.run([probs_1,concat_layer_result, summary_op], feed_dict= {x_1 : img_list, x_2:img_list})
tf_wirter.add_summary(summary, 1)
print("End time : %.5ss" % (time.time() - start_time))

# print_prob(prob[0][1:]) # Note : as it have 1001 outputs, the 1st output is nothing
tf_wirter.close()

prob = prob[0]
print("  End time : %.5ss" % (time.time() - start_time))
preds = (np.argsort(prob)[::-1])[0:5]
print(preds)
for p in preds:
    print(p, prob[p])
print(concat_result.shape)

