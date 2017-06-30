import numpy as np
import tensorflow as tf

EPOCHS = 10000
PRINT_STEP = 1000

data_train = np.array([[1, 2, 3, 4, 5], [2, 3, 4, 5, 6], [3, 4, 5, 6, 7]])
print(data_train.shape)  ## (3,5)
target_train = np.array([[6], [7], [8]])

x_ = tf.placeholder(tf.float32, [None, data_train.shape[1]])
y_ = tf.placeholder(tf.float32, [None, 1])

cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=data_train.shape[1])

outputs, states = tf.nn.static_rnn(cell, [x_], dtype=tf.float32)
outputs = outputs[-1]

W = tf.Variable(tf.random_normal([data_train.shape[1], 1]))
b = tf.Variable(tf.random_normal([1]))

y = tf.matmul(outputs, W) + b

cost = tf.reduce_mean(tf.square(y - y_))
train_op = tf.train.RMSPropOptimizer(0.005, 0.2 ).minimize(cost)


with tf.Session() as sess:
    tf.initialize_all_variables().run()
    for i in range(EPOCHS):
        states_value, _ = sess.run([states, train_op], feed_dict={x_: data_train, y_: target_train})
        if i == 10:
            print(states_value)
            print(states_value.c.shape)
        if i % PRINT_STEP == 0:
            c = sess.run(cost, feed_dict={x_: data_train, y_: target_train})
            print('training cost:', c)

    response = sess.run(y, feed_dict={x_: data_train})
    print(response)
