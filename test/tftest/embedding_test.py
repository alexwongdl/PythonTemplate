"""
Created by Alex Wang on 2017-06-23
测试tf.nn.embedding_lookup
"""
import tensorflow as tf
import numpy as np

def test_embedding_lookup():
    """
    测试tf.nn.embedding_lookup
    :return:
    """
    matrix = tf.Variable(tf.random_uniform([100, 20],0,1 ), name="dict")
    ids = np.array([1,4,5])
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    x = tf.nn.embedding_lookup(matrix, ids)
    x_value = sess.run([x])
    print(x_value)
    # tf.nn.softmax()

if __name__ == "__main__":
    test_embedding_lookup()