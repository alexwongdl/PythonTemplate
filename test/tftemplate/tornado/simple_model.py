import tensorflow as tf
import numpy as np

tf.reset_default_graph()
a = tf.placeholder(tf.int32, shape=(), name="input")
b = tf.get_variable("b", shape=(), dtype=tf.int32)
asquare = tf.multiply(a, a, name="output")

sess = tf.Session()
sess.run(tf.global_variables_initializer())
print(sess.run([asquare], feed_dict={a: 2}))

saver = tf.train.Saver()
save_path = saver.save(sess, "/home/recsys/hzwangjian1/data/testsaver.model")
print(save_path)
