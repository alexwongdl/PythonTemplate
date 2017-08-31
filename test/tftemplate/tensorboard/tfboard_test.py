"""
Created by Alex Wang on 2017-06-25
"""
import tensorflow as tf

def test_board():
    tf.reset_default_graph()
    sess = tf.Session()

    constant_one = tf.Variable([1.0, 1.0], name="constant_one")
    var_one = tf.Variable(initial_value=[1.0, 2.0], name="var_one")
    summary_writer = tf.summary.FileWriter("E://workspace/python/tensorboard/log", sess.graph)   ####TODO: WARN 必须要加sess.graph

    tf.summary.tensor_summary(var_one.op.name, var_one)
    summary_op = tf.summary.merge_all()
    sess.run(tf.global_variables_initializer())

    for i in range(10):
        var_one = tf.add(var_one, constant_one)
        var_one_val, summary = sess.run([var_one, summary_op])
        print(var_one_val)
        summary_writer.add_summary(summary, i)
    summary_writer.close()
    sess.close()

def test_board_simple():
    tf.reset_default_graph()
    sess = tf.Session()


    summary_writer = tf.summary.FileWriter("E://workspace/python/tensorboard/log", sess.graph)

    with tf.name_scope("total") as scope:
        constant_one = tf.Variable(1.0, name="constant_one")
        var_one = tf.Variable(initial_value=1.0, name="var_one")
        var_two = tf.add(constant_one, var_one, name="var_two")
        tf.summary.scalar(var_two.op.name, var_two)

    summary_op = tf.summary.merge_all()
    sess.run(tf.global_variables_initializer())

    var_two_val, summary = sess.run([var_two, summary_op])
    print(var_two_val)
    summary_writer.add_summary(summary, 0)
    summary_writer.close()
    sess.close()


if __name__ == "__main__":
    test_board_simple()