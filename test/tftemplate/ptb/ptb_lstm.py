"""
Created by Alex Wang on 20170706
"""
import tensorflow as tf
import numpy as np
# from ptb_reader import ptb_raw_data
from test.tftemplate.ptb.ptb_reader import ptb_raw_data
from test.tftemplate.ptb.ptb_reader import ptb_data_queue
# from ptb_reader import ptb_raw_data
# from ptb_reader import ptb_data_queue
from  tfutil import data_batch_fetch

params = {
    "init_scale": 0.1,  # the initial scale of the weights
    "learning_rate": 1.0,  # the initial value of the learning rate
    "max_grad_norm": 5,  # the maximum permissible norm of the gradient
    "num_layers": 2,  # the number of LSTM layers
    "num_steps": 20,  # the number of unrolled steps of LSTM
    "hidden_size": 200,  # the number of LSTM units
    "max_epoch": 4,  # the number of epochs trained with the initial learning rate
    "max_max_epoch": 13,  # the total number of epochs for training
    "keep_prob": 1.0,  # the probability of keeping weights in the dropout layer
    "lr_decay": 0.5,  # the decay of the learning rate for each epoch after "max_epoch"
    "batch_size": 20,  # the batch size
    "vocab_size": 10000
}


def createLSTMCell(keep_prob):
    cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=params["hidden_size"], forget_bias=0.0)
    return tf.nn.rnn_cell.DropoutWrapper(cell, keep_prob)  ## lstm输出层dropout


class PTBModel():
    def __init__(self):
        self.keep_prob = tf.placeholder(tf.float32, shape=(), name="keep_prob")
        self.input_x = tf.placeholder(tf.int32, shape=(params["batch_size"], params["num_steps"]), name="input_x")
        self.input_y = tf.placeholder(tf.int32, shape=(params["batch_size"], params["num_steps"]), name="input_y")

        self.network = tf.nn.rnn_cell.MultiRNNCell(
                [createLSTMCell(self.keep_prob) for i in range(params["num_layers"])])
        self.net_init_state = self.network.zero_state(params["batch_size"], tf.float32)
        initializer = tf.random_uniform_initializer(-params["init_scale"], params["init_scale"])

        with tf.device("/cpu:0"):
            ## 输入id嵌入到字典
            embedding_dict = tf.get_variable("embedding_dict", [params["vocab_size"], params["hidden_size"]],
                                             dtype=tf.float32, initializer= initializer)

        self.input_x_embed = tf.nn.embedding_lookup(embedding_dict, self.input_x)
        self.input_x_embed = tf.nn.dropout(self.input_x_embed, self.keep_prob)  ## 输入层dropout
        self.outputs = []
        state = self.net_init_state
        with tf.variable_scope("RNN"):
            for step in range(params["num_steps"]):
                if step > 0:
                    tf.get_variable_scope().reuse_variables()
                (cell_output, state) = self.network(self.input_x_embed[:, step, :], state)  ## batch_size * d
                self.outputs.append(cell_output)  ## num_steps * batch_size * d

        ## TODO:`N` of tensors of shape `(A, B, C)` --> `(A, N, B, C)` batch_size * num_steps * d
        self.output = tf.reshape(tf.stack(axis=1, values=self.outputs), [-1, params["hidden_size"]])
        softmax_w = tf.get_variable("softmax_w", [params["hidden_size"], params["vocab_size"]], dtype=tf.float32, initializer=initializer)
        softmax_b = tf.get_variable("softmax_b", [1, params["vocab_size"]], dtype=tf.float32, initializer=initializer)

        self.logits = tf.matmul(self.output, softmax_w) + softmax_b
        self.logits = tf.reshape(self.logits, [params["batch_size"], params["num_steps"], params["vocab_size"]])

        self.loss = tf.contrib.seq2seq.sequence_loss(self.logits, self.input_y,
                                                     tf.ones([params["batch_size"], params["num_steps"]], tf.float32),
                                                     average_across_timesteps=False)  ## TODO:?

        self.predict = tf.arg_max(self.logits, 2)
        self.cost = tf.reduce_sum(self.loss)

        ## 梯度操作
        self._lr = tf.placeholder(tf.float32, shape=[], name="learning_rate")
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), params["max_grad_norm"])
        optimizer = tf.train.GradientDescentOptimizer(self._lr)
        self._train_op = optimizer.apply_gradients(zip(grads, tvars),
                                                   global_step=tf.contrib.framework.get_or_create_global_step())


def main():
    data_path = "E://data/ptb/data"
    # data_path = "/home/recsys/hzwangjian1/learntf/ptb_data"
    train_data, test_data, valid_data, word_to_id, id_to_word = ptb_raw_data(data_path)
    sess = tf.Session()
    model = PTBModel()
    x, y = ptb_data_queue(train_data, batch_size=params["batch_size"], num_steps=params["num_steps"])

    sess.run(tf.global_variables_initializer())
    coord, threads = data_batch_fetch.start_queue_runner(sess)

    total_steps = 100000
    for i in range(total_steps):
        # lr_decay = 1.0 ** max(i + 1 - params["max_max_epoch"], 0.0)
        lr_decay = 1.0 * 0.99 ** (i / 1000)
        x_value, y_value = sess.run([x, y])
        # if i % 1000 == 0:
        # for i in range(len(x_value)):
        #     x_words = [id_to_word[id] for id in x_value[i] if id in id_to_word]
        #     print("x_words:" + " ".join(x_words))
        #     y_words = [id_to_word[id] for id in y_value[i] if id in id_to_word]
        #     print("y_words:" + " ".join(y_words))

        cost_value, predict, train_op = sess.run([model.cost, model.predict, model._train_op], feed_dict={model.keep_prob: params["keep_prob"],
                                                                               model.input_x: x_value,
                                                                               model.input_y: y_value,
                                                                               model._lr: lr_decay})
        if i % 1000 == 0:
            print("round :" + str(i))
            print(len(x_value))
            print("epoch:{}\tcost_value:{}".format(i, cost_value))

            for k in range(len(predict)):
                print(predict[k])
                print(y_value[k])

    data_batch_fetch.stop_queue_runner(coord, threads)
    sess.close()


if __name__ == "__main__":
    main()
