import numpy as np
import tensorflow as tf

EPOCHS = 10000
PRINT_STEP = 1000

data_train = np.array([[1, 2, 3, 4, 5], [2, 3, 4, 5, 6], [3, 4, 5, 6, 7]])
print(data_train.shape)  ## (3,5)
target_train = np.array([[6], [7], [8]])

x_ = tf.placeholder(tf.float32, [None, data_train.shape[1]])
y_ = tf.placeholder(tf.float32, [None, 1])

## 新建两个LSTM单元
cell_one = tf.nn.rnn_cell.BasicLSTMCell(num_units=data_train.shape[1])
cell_two = tf.nn.rnn_cell.BasicLSTMCell(num_units=data_train.shape[1])

## LSTM叠加构建网络
networks = tf.nn.rnn_cell.MultiRNNCell([cell_one, cell_two], state_is_tuple=True)
init_state = networks.zero_state(data_train.shape[0], dtype=tf.float32)
outputs, states = networks.call(x_, init_state)  ## outputs:[batch_size, d]  states:(cell_num) * LSTMStateTuple

W = tf.Variable(tf.random_normal([data_train.shape[1], 1]))
b = tf.Variable(tf.random_normal([1]))

y = tf.matmul(outputs, W) + b

cost = tf.reduce_mean(tf.square(y - y_))
train_op = tf.train.RMSPropOptimizer(0.005, 0.2).minimize(cost)

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    for i in range(EPOCHS):
        outputs_value, states_value, _ = sess.run([outputs, states, train_op],
                                                  feed_dict={x_: data_train, y_: target_train})
        if i == 10:
            print("outputs:" + str(outputs_value))
            print(states_value)
            # print(states_value.c.shape)
        if i % PRINT_STEP == 0:
            c = sess.run(cost, feed_dict={x_: data_train, y_: target_train})
            print('training cost:', c)

    response = sess.run(y, feed_dict={x_: data_train})
    print(response)
    response = sess.run(y, feed_dict={x_: [[4, 5, 6, 7, 8], [5, 6, 7, 8, 9], [6, 7, 8, 9, 10]]})
    print(response)
