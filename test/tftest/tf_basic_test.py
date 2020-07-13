"""
Created by Alex Wang on 2020-01-30
测试tensorflow的基本操作op
"""
import tensorflow as tf
import numpy as np


def test_padding():
    """
    https://stackoverflow.com/questions/42334646/tensorflow-pad-unknown-size-tensor-to-a-specific-size
    https://stackoverflow.com/questions/43928642/how-does-tensorflow-pad-work/52204985
    'paddings' is [[1, 1,], [2, 2]]. Try to map this vale us as [[top,bottom],[left,right]]. i.e.

    top = 1,      //Extra padding introduce on top
    bottom = 1,   //Extra padding introduce on bottom
    left = 2,     //Extra padding introduce on left
    right = 2.    //Extra padding introduce on right
    Try another example where 'padding' is [[2, 1], [2, 3]]. Output will be:

    [[0 0 0 0 0 0 0 0]
     [0 0 0 0 0 0 0 0]
     [0 0 1 2 3 0 0 0]
     [0 0 4 5 6 0 0 0]
     [0 0 0 0 0 0 0 0]]
    Here top=2, bottom=1, left=2, right=3.
    :return:
    """
    # 未知形状的tensor进行padding
    a = tf.constant(np.random.rand(5, 20))
    paddings = [[0, 16 - tf.shape(a)[0]], [0, 0]]
    out = tf.pad(a, paddings, 'CONSTANT', constant_values=0)

    with tf.Session() as sess:
        out_val = sess.run(out)
    print('shape of out_val:{}'.format(out_val.shape))
    print('out_val:{}'.format(out_val))


if __name__ == '__main__':
    test_padding()
