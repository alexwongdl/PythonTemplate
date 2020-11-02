"""
Created by Alex Wang
text:w2v + transformer(masked)

frame: CNN + NeXtVLAD
SE CG
OCR
"""

import os
from datetime import datetime
import traceback

import numpy as np
import random
import time
import tensorlayer as tl
import tensorflow as tf
import tensorflow.contrib.slim as slim

import utils.eval_util
from utils.attention_layer import attention_layer, dropout, create_initializer
from utils.eval_util import EvaluationMetrics
from utils.util import cal_accuracy, one_hot_encode, top_n_accuracy
from utils.layer_unit import se_context_gate, moe_model
from utils.layer_unit import NeXtVLAD
from utils.badcase_analysis import badcase_set_get
from utils.util import load_id_name_map

tf.app.flags.DEFINE_string('tables', '', 'table_list')
tf.app.flags.DEFINE_integer('task_index', -1, 'worker task index')
tf.app.flags.DEFINE_string('ps_hosts', "", "ps hosts")
tf.app.flags.DEFINE_string('worker_hosts', "", "worker hosts")
tf.app.flags.DEFINE_string('job_name', "", "job name:worker or ps")
tf.app.flags.DEFINE_string("buckets", None, "oss buckets")
tf.app.flags.DEFINE_integer("batch_size", 32, "batch size")
tf.app.flags.DEFINE_float("drop_rate", 0.5, "drop_rate")
tf.app.flags.DEFINE_float("learning_rate", 0.0001, "init learning rate")
tf.app.flags.DEFINE_float("frame_weight", 0.0001, "init learning rate")
tf.app.flags.DEFINE_float("text_weight", 0.001, "init learning rate")  # better than 0.0001
tf.app.flags.DEFINE_float("w2v_weight", 0.000001, "init learning rate")
tf.app.flags.DEFINE_float("root_weight", 1.0, "init learning rate")

tf.app.flags.DEFINE_boolean("train_w2v", False, "modify word embedding or not")
tf.app.flags.DEFINE_boolean("frame_aug", False, "augment frame data or not, if True, use frame_feat_process_augment")
tf.app.flags.DEFINE_boolean("text_aug", False, "augment text data or not, if True, use text_fea_process_augment")
tf.app.flags.DEFINE_float("drop_text_rate", 0.05, "rate of drop title or summary")
tf.app.flags.DEFINE_float("drop_word_rate", 0.05, "rate of drop word in title or summary")
tf.app.flags.DEFINE_float("drop_item_rate", 0.5, "rate of drop word in title or summary")

tf.app.flags.DEFINE_float("decay_rate", 0.95, "learning rate decay rate")
tf.app.flags.DEFINE_float("decay_step", 3000, "learning rate decay step")
tf.app.flags.DEFINE_boolean("use_cg", False, "use context gate or not")

#  NeXtVLAD
tf.app.flags.DEFINE_integer("nextvlad_cluster_size", 256, "Number of units in the NeXtVLAD cluster layer.")
tf.app.flags.DEFINE_integer("nextvlad_hidden_size", 1024, "Number of units in the NeXtVLAD hidden layer.")

tf.app.flags.DEFINE_integer("groups", 8, "number of groups in VLAD encoding")
tf.app.flags.DEFINE_float("vlad_drop_rate", 0.5, "dropout ratio after VLAD encoding")
tf.app.flags.DEFINE_integer("expansion", 2, "expansion ratio in Group NetVlad")
tf.app.flags.DEFINE_integer("gating_reduction", 2, "reduction factor in se context gating")

# attention
tf.app.flags.DEFINE_integer("attention_embed_dim", 256, "attention embedding dimension")
tf.app.flags.DEFINE_integer("attention_layer_num", 5, "attention layer number")

# text
tf.app.flags.DEFINE_integer("embed_size", 200, "word embedding size")

# labe num
tf.app.flags.DEFINE_integer("valid_batch_num", 17, "batch num of valid, valid examples num: valid_batch_num * 32")
tf.app.flags.DEFINE_integer("label_1_num", 89, " ")
tf.app.flags.DEFINE_integer("label_2_num", 33, " ")

tf.app.flags.DEFINE_string("restore_name", None, "restore checkpoint")

FLAGS = tf.app.flags.FLAGS

print('tables:', FLAGS.tables)
print('task_index:', FLAGS.task_index)
print('ps_hosts', FLAGS.ps_hosts)
print('worker_hosts', FLAGS.worker_hosts)
print('job_name', FLAGS.job_name)

# parameters
batch_size = FLAGS.batch_size
# TODO
drop_rate = FLAGS.drop_rate
LABEL_1_NUM = FLAGS.label_1_num
LABEL_2_NUM = FLAGS.label_2_num
valid_batch_num = FLAGS.valid_batch_num  # 32*17=544 -- 538  32*40=1280 -- 1271
ATTENTION_EMBED_DIM = FLAGS.attention_embed_dim

# text
vocab_size = 196160
title_length = 50
desc_length = 100
item_title_length = 50
item_summary_length = 100
item_cate_length = 50

text_length = 150
num_filter = 256
embed_size = FLAGS.embed_size
_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5

# text
# vocab_size = 196160
# title_length = 50
# desc_length = 100
# num_filter = 256
# _BATCH_NORM_DECAY = 0.997
# _BATCH_NORM_EPSILON = 1e-5

tables = FLAGS.tables.split(",")
train_table = tables[0]
valid_table = tables[1]
word2vec_table = tables[2]
print('train_table:{}'.format(train_table))
print('valid_table:{}'.format(valid_table))
print('word2vec_table:{}'.format(word2vec_table))

ps_spec = FLAGS.ps_hosts.split(",")
worker_spec = FLAGS.worker_hosts.split(",")
cluster = tf.train.ClusterSpec({"ps": ps_spec, "worker": worker_spec})
worker_count = len(worker_spec)
task_index = int(FLAGS.task_index)

is_chief = task_index == 0  # regard worker with index 0 as chief
print('is chief:', is_chief)

config = tf.ConfigProto(intra_op_parallelism_threads=48)
server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=task_index)
if FLAGS.job_name == "ps":
    server.join()

worker_device = "/job:worker/task:%d/cpu:%d" % (task_index, 0)
print("worker device:", worker_device)


def time_print(string):
    print('[{}] {}'.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), string))


def load_embedding(file_name, vocab_size, embedding_size, vectors, shift=0, name='waou'):
    with tf.device("/cpu:0"):
        with tf.name_scope('load_w2v_embed_' + name):
            reader = tf.TableRecordReader(
                selected_cols='word,vector',
                csv_delimiter=',',
                name=name)
            file_queue = tf.train.string_input_producer([file_name], name='w2v_queue_' + name)
            _, values = reader.read_up_to(file_queue, vocab_size, name='w2v_read_' + name)
            embed_raw = tf.decode_csv(
                values, record_defaults=[[''] for _ in range(1 + embedding_size)], field_delim=',')
            embed_raw = tf.transpose(embed_raw)
            ids = tf.string_to_number(embed_raw[:, 0], tf.int32, name='word_ids_' + name)
            ids = tf.reshape(ids, [-1])
            embeddings = tf.string_to_number(embed_raw[:, 1:1 + embedding_size], tf.float32)
            init = tf.scatter_update(vectors, ids + shift, embeddings, name='word_ids_scatter_update' + name).op
    return init


# 1. load data
with tf.device(worker_device):
    # filename_queue = tf.train.string_input_producer([FLAGS.tables], num_epochs=100000000)
    dataset = tf.data.TableRecordDataset([train_table],
                                         record_defaults=(np.int64(1), '', -1, -1, '', '', '', '', ''),
                                         selected_cols="video_id,vdo_feature,label1,label2,title_code,desc_code,i_title_code,item_summary_code,cate_str_code",
                                         slice_count=worker_count,
                                         slice_id=task_index,
                                         num_threads=64,
                                         capacity=512)

    dataset = dataset.batch(batch_size)
    if FLAGS.frame_aug:
        dataset = dataset.map(frame_augment_preprocess)
    dataset = dataset.repeat(100000000)
    # dataset = dataset.shuffle(200) # TODO:
    iterator = dataset.make_initializable_iterator()
    next_elems = iterator.get_next()
    (video_id_batch, frame_fea_batch, label_1_batch, label_2_batch, title_batch, desc_batch, item_title_batch,
     item_summary_batch, item_cate_batch) = next_elems

    # valid dataset
    valid_dataset = tf.data.TableRecordDataset([valid_table],
                                               record_defaults=(np.int64(1), '', -1, -1, '', '', '', '', ''),
                                               selected_cols="video_id,vdo_feature,label1,label2,title_code,desc_code,i_title_code,item_summary_code,cate_str_code",
                                               # slice_count=worker_count,
                                               # slice_id=task_index,
                                               num_threads=32,
                                               capacity=256)
    valid_dataset = valid_dataset.batch(batch_size)
    valid_dataset = valid_dataset.repeat(100000000)
    valid_iterator = valid_dataset.make_initializable_iterator()
    (valid_video_id, valid_frame_fea, valid_label_1, valid_label_2, valid_title, valid_desc, valid_item_title,
     valid_item_summary, valid_item_cate) = valid_iterator.get_next()

# 2. construct network
available_worker_device = "/job:worker/task:%d" % (task_index)
with tf.device(tf.train.replica_device_setter(worker_device=available_worker_device, cluster=cluster)):
    global_step = tf.Variable(0, name="global_step", trainable=False)


def construct_network(frame_input, label_1, label_2, reuse, is_training, title_input, desc_input,
                      item_title_input, item_summary_input, item_cate_input, word_embed):
    """
    :param frame_input:
    :param tags_input:
    :param reuse:
    :param is_training:
    :return:
    """
    with tf.variable_scope('text', reuse=reuse) as scope:
        with tf.device("/cpu:0"), tf.variable_scope('dict'):
            # word_embedding = tf.get_variable('initW', [vocab_size, embed_size], trainable=True)
            title_raw = tf.nn.embedding_lookup(word_embed, title_input)
            desc_raw = tf.nn.embedding_lookup(word_embed, desc_input)
            item_title_raw = tf.nn.embedding_lookup(word_embed, item_title_input)
            cate_raw = tf.nn.embedding_lookup(word_embed, item_cate_input)

        with tf.variable_scope("conv"):
            def txt_conv(t_input, d_input, conv_w, name):
                text = tf.concat([t_input, d_input], axis=1)
                conv = tf.layers.conv1d(text, filters=num_filter, kernel_size=conv_w, name=name)
                conv = slim.batch_norm(conv,
                                       decay=0.9997,
                                       epsilon=0.001,
                                       is_training=is_training)
                conv = tf.reduce_max(conv, reduction_indices=[1], name='global_pool_title_desc')
                return conv

            rep_2 = txt_conv(title_raw, desc_raw, 2, 'conv2')
            rep_3 = txt_conv(title_raw, desc_raw, 3, 'conv3')
            rep_4 = txt_conv(title_raw, desc_raw, 4, 'conv4')
            rep_5 = txt_conv(title_raw, desc_raw, 5, 'conv5')

            rep_cate_2 = txt_conv(cate_raw, item_title_raw, 2, 'conv2_1')
            rep_cate_3 = txt_conv(cate_raw, item_title_raw, 2, 'conv3_1')
            rep_cate_4 = txt_conv(cate_raw, item_title_raw, 2, 'conv4_1')
            rep_cate_5 = txt_conv(cate_raw, item_title_raw, 2, 'conv5_1')

            rep = tf.concat([rep_2, rep_3, rep_4, rep_5], 1)
            rep_cate = tf.concat([rep_cate_2, rep_cate_3, rep_cate_4, rep_cate_5], 1)

            text_logits_1 = tf.layers.dense(rep, 256)  # 512
            text_logits_2 = tf.layers.dense(rep_cate, 256)
            text_logits = tf.concat([text_logits_1, text_logits_2], 1)

        with tf.variable_scope("transformer"):
            with tf.variable_scope('preprocess', reuse=reuse) as scope:
                frame_position_embeddings = tf.get_variable(
                    name='frame_position_embedding',
                    shape=[text_length, ATTENTION_EMBED_DIM],
                    initializer=tf.truncated_normal_initializer(stddev=0.02))
                frame_parts = tf.layers.conv1d(tf.concat([title_raw, desc_raw], axis=1), filters=ATTENTION_EMBED_DIM,
                                               kernel_size=1, name='frame_feat_squeeze')
                frame_parts = slim.batch_norm(frame_parts,
                                              decay=0.9997,
                                              epsilon=0.001,
                                              is_training=is_training)
                frame_parts += frame_position_embeddings

            intermediate_size = 512
            hidden_size = ATTENTION_EMBED_DIM
            initializer_range = 0.02
            hidden_dropout_prob = 0.2

            prev_output = frame_parts
            for layer_idx in range(FLAGS.attention_layer_num):
                with tf.variable_scope("layer_%d" % layer_idx):
                    layer_input = prev_output

                    with tf.variable_scope("attention"):
                        with tf.variable_scope("self"):
                            attention_head = attention_layer(
                                from_tensor=layer_input,
                                to_tensor=layer_input,
                                is_training=is_training,
                                attention_mask=None,
                                num_attention_heads=4,
                                size_per_head=64,
                                attention_probs_dropout_prob=hidden_dropout_prob,
                                # if modified, should pay attention to inference
                                initializer_range=0.02,
                                do_return_2d_tensor=False,
                                batch_size=batch_size,
                                from_seq_length=text_length,
                                to_seq_length=text_length)

                            attention_output = attention_head

                        # Run a linear projection of `hidden_size` then add a residual
                        # with `layer_input`.
                        with tf.variable_scope("output"):
                            attention_output = tf.layers.dense(
                                attention_output,
                                hidden_size,
                                kernel_initializer=create_initializer(initializer_range))
                            attention_output = dropout(attention_output, hidden_dropout_prob, is_training=is_training)
                            attention_output = slim.batch_norm(attention_output + layer_input,
                                                               decay=0.9997,
                                                               epsilon=0.001,
                                                               is_training=is_training)

                    # The activation is only applied to the "intermediate" hidden layer.
                    with tf.variable_scope("intermediate"):
                        intermediate_output = tf.layers.dense(
                            attention_output,
                            intermediate_size,
                            activation=tf.nn.relu,
                            kernel_initializer=create_initializer(initializer_range))

                    # Down-project back to `hidden_size` then add the residual.
                    with tf.variable_scope("output"):
                        layer_output = tf.layers.dense(
                            intermediate_output,
                            hidden_size,
                            kernel_initializer=create_initializer(initializer_range))
                        layer_output = dropout(layer_output, hidden_dropout_prob, is_training=is_training)
                        layer_output = slim.batch_norm(layer_output + attention_output,
                                                       decay=0.9997,
                                                       epsilon=0.001,
                                                       is_training=is_training)
                        prev_output = layer_output

            attention_final = tf.reduce_max(prev_output, [1], keep_dims=False,
                                            name='reduce_max')  # 256

    with tf.variable_scope('NeXtVLAD', reuse=reuse) as scope:
        # re_d = 512
        # frame_input_1 = tf.layers.dense(frame_input, re_d, activation=tf.nn.relu, name='re_d')

        video_nextvlad = NeXtVLAD(FRAME_FEAT_DIM, FRAME_FEAT_LEN,
                                  FLAGS.nextvlad_cluster_size, is_training,
                                  groups=FLAGS.groups, expansion=FLAGS.expansion)

        vlad = video_nextvlad.forward(frame_input, mask=None)

        # vlad = slim.dropout(vlad, keep_prob=1. - FLAGS.vlad_drop_rate, is_training=is_training, scope="vlad_dropout")
        #
        # # SE context gating
        # vlad_dim = vlad.get_shape().as_list()[1]
        # # print("VLAD dimension", vlad_dim)
        # hidden1_weights = tf.get_variable("hidden1_weights",
        #                                   [vlad_dim, FLAGS.nextvlad_hidden_size],
        #                                   initializer=slim.variance_scaling_initializer())
        #
        # activation = tf.matmul(vlad, hidden1_weights)
        # activation = slim.batch_norm(
        #     activation,
        #     center=True,
        #     scale=True,
        #     is_training=is_training,
        #     scope="hidden1_bn",
        #     fused=False)
        #
        # # activation = tf.nn.relu(activation)
        #
        # gating_weights_1 = tf.get_variable("gating_weights_1",
        #                                    [FLAGS.nextvlad_hidden_size,
        #                                     FLAGS.nextvlad_hidden_size // FLAGS.gating_reduction],
        #                                    initializer=slim.variance_scaling_initializer())
        #
        # gates = tf.matmul(activation, gating_weights_1)
        #
        # gates = slim.batch_norm(
        #     gates,
        #     center=True,
        #     scale=True,
        #     is_training=is_training,
        #     activation_fn=slim.nn.relu,
        #     scope="gating_bn")
        #
        # gating_weights_2 = tf.get_variable("gating_weights_2",
        #                                    [FLAGS.nextvlad_hidden_size // FLAGS.gating_reduction,
        #                                     FLAGS.nextvlad_hidden_size],
        #                                    initializer=slim.variance_scaling_initializer()
        #                                    )
        # gates = tf.matmul(gates, gating_weights_2)
        #
        # gates = tf.sigmoid(gates)
        #
        # vlad_activation = tf.multiply(activation, gates)
        vlad_activation = tf.layers.dense(vlad, 512, name='fc1')  # 512

    with tf.variable_scope('frame', reuse=reuse) as scope:
        # layer 1 (batch * 200 * 1024)
        nets_frame = tf.layers.conv1d(frame_input, filters=1024, kernel_size=3, name='conv1d_1')
        nets_frame = slim.batch_norm(nets_frame,
                                     decay=0.9997,
                                     epsilon=0.001,
                                     is_training=is_training)
        nets_frame = tf.nn.relu(nets_frame)
        nets_frame = tf.layers.max_pooling1d(nets_frame, pool_size=2, strides=2, name='pool1d_1')

        # layer 2
        nets_frame = tf.layers.conv1d(nets_frame, filters=256, kernel_size=5, name='conv1d_2')
        nets_frame = slim.batch_norm(nets_frame,
                                     decay=0.9997,
                                     epsilon=0.001,
                                     is_training=is_training)
        nets_frame = tf.nn.relu(nets_frame)
        # layer 3
        nets_frame = tf.layers.conv1d(nets_frame, filters=256, kernel_size=5, name='conv1d_3')
        nets_frame = slim.batch_norm(nets_frame,
                                     decay=0.9997,
                                     epsilon=0.001,
                                     is_training=is_training)
        nets_frame = tf.nn.relu(nets_frame)  # 91 * 256
        # max pooling layer
        nets_frame = tf.layers.max_pooling1d(nets_frame, pool_size=4, strides=4, name='pool1d_2')

        # test flat
        nets_frame = tf.layers.flatten(nets_frame)  # 5632 = 22 * 256
        # nets_frame = tf.reduce_max(nets_frame, reduction_indices=[1], name='max_pool')

        fc_frame = tf.layers.dense(nets_frame, 512, name='fc1')  # 512
        # fc_frame = tf.nn.l2_normalize(fc_frame, dim=1)

    with tf.variable_scope('predict', reuse=reuse) as scope:
        # video_vector = tf.concat([fc_frame, text_logits, attention_final, vlad_activation], axis=1)  # 1280

        frame_concat = tf.concat([fc_frame, vlad_activation], axis=1)
        text_concat = tf.concat([text_logits, attention_final], axis=1)

        frame_vector = tf.layers.dense(frame_concat, 1024, name='frame_dense')
        frame_vector = tf.layers.dropout(frame_vector, drop_rate, training=is_training)
        frame_vector = tf.nn.leaky_relu(frame_vector)
        frame_vector = tf.layers.dense(frame_vector, 512, name='frame_dense_1')
        frame_vector = slim.batch_norm(frame_vector,
                                       decay=0.9997,
                                       epsilon=0.001,
                                       is_training=is_training)

        text_vector = tf.layers.dense(text_concat, 1024, name='text_dense')
        text_vector = tf.layers.dropout(text_vector, drop_rate, training=is_training)
        text_vector = tf.nn.leaky_relu(text_vector)
        text_vector = tf.layers.dense(text_vector, 512, name="text_dense_1")
        text_vector = slim.batch_norm(text_vector,
                                      decay=0.9997,
                                      epsilon=0.001,
                                      is_training=is_training)

        with tf.variable_scope('text_predict', reuse=reuse) as scope:
            predict_text_1 = tf.layers.dense(text_vector, LABEL_1_NUM, name='pred_text_1')
            predict_text_2 = tf.layers.dense(text_vector, LABEL_2_NUM, name='pred_text_2')
            cross_entropy_text_1 = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=predict_text_1, labels=label_1)
            loss_text_1 = tf.reduce_mean(cross_entropy_text_1)

            cross_entropy_text_2 = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=predict_text_2, labels=label_2)
            loss_text_2 = tf.reduce_mean(cross_entropy_text_2)

        with tf.variable_scope('frame_predict', reuse=reuse) as scope:
            predict_frame_1 = tf.layers.dense(frame_vector, LABEL_1_NUM, name='pred_frame_1')
            predict_frame_2 = tf.layers.dense(frame_vector, LABEL_2_NUM, name='pred_frame_2')
            cross_entropy_frame_1 = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=predict_frame_1,
                                                                                   labels=label_1)
            loss_frame_1 = tf.reduce_mean(cross_entropy_frame_1)

            cross_entropy_frame_2 = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=predict_frame_2,
                                                                                   labels=label_2)
            loss_frame_2 = tf.reduce_mean(cross_entropy_frame_2)

        video_vector = tf.concat([frame_vector, text_vector], axis=1)
        total_vector = slim.batch_norm(video_vector,
                                       decay=0.9997,
                                       epsilon=0.001,
                                       is_training=is_training)

        # -- root predict
        with tf.variable_scope('root_se_cg', reuse=reuse) as scope:
            root_vector = se_context_gate(total_vector, is_training=is_training, se_hidden_size=512)
        # tf.check_numerics(root_vector, 'root_vector is inf or nan')
        with tf.variable_scope('label_1', reuse=reuse) as scope:
            predict_root_1 = tf.layers.dense(root_vector, LABEL_1_NUM, name='pred_root_1')
            predict_root_label_1 = tf.argmax(predict_root_1, dimension=-1)
            predict_root_confidence_1 = tf.nn.softmax(predict_root_1, name='conf_root_1')

            cross_entropy_1 = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=predict_root_1, labels=label_1)
            loss_root_1 = tf.reduce_mean(cross_entropy_1)

        with tf.variable_scope('label_2', reuse=reuse) as scope:
            predict_root_2 = tf.layers.dense(root_vector, LABEL_2_NUM, name='pred_root_2')
            predict_root_label_2 = tf.argmax(predict_root_2, dimension=-1)
            predict_root_confidence_2 = tf.nn.softmax(predict_root_2, name='conf_root_2')

            cross_entropy_2 = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=predict_root_2, labels=label_2)
            loss_root_2 = tf.reduce_mean(cross_entropy_2)

        loss_root = loss_root_1 + loss_root_2
        # https://stackoverflow.com/questions/44560549/unbalanced-data-and-weighted-cross-entropy
        # tf.losses.sparse_softmax_cross_entropy()

    with tf.variable_scope('l2_norm', reuse=reuse) as scope:
        L2_frame = tf.Variable(initial_value=0., trainable=False, dtype=tf.float32)
        L2_text = tf.Variable(initial_value=0., trainable=False, dtype=tf.float32)
        L2_w2v = tf.Variable(initial_value=0., trainable=False, dtype=tf.float32)
        for w in tl.layers.get_variables_with_name('frame', True, True):
            L2_frame += tf.contrib.layers.l2_regularizer(1.0)(w)

        for w in tl.layers.get_variables_with_name('predict', True, True):
            L2_frame += tf.contrib.layers.l2_regularizer(1.0)(w)

        for w in tl.layers.get_variables_with_name('NeXtVLAD', True, True):
            L2_frame += tf.contrib.layers.l2_regularizer(1.0)(w)

        for w in tl.layers.get_variables_with_name('text', True, True):
            L2_text += tf.contrib.layers.l2_regularizer(1.0)(w)

        if FLAGS.train_w2v:
            for w in tl.layers.get_variables_with_name('initW', True, True):
                L2_w2v += tf.contrib.layers.l2_regularizer(1.0)(w)

    cost = FLAGS.root_weight * loss_root + FLAGS.frame_weight * L2_frame + \
           FLAGS.text_weight * L2_text + FLAGS.w2v_weight * L2_w2v + \
            loss_text_1 + loss_text_2 + loss_frame_1 + loss_frame_2

    # loss_root_1_summary = tf.summary.scalar('loss_root_1', loss_root_1)
    # loss_root_2_summary = tf.summary.scalar('loss_root_2', loss_root_2)
    # summary_op = tf.summary.merge([loss_root_1_summary, loss_root_2_summary])

    result = dict()
    result['loss_root_1'] = loss_root_1
    result['loss_root_2'] = loss_root_2
    result['cost'] = cost
    result['predict_root_1'] = predict_root_1
    result['predict_label_root_1'] = predict_root_label_1
    result['confidence_root_1'] = predict_root_confidence_1
    result['predict_root_2'] = predict_root_2
    result['predict_label_root_2'] = predict_root_label_2
    result['confidence_root_2'] = predict_root_confidence_2
    result['L2_frame'] = L2_frame
    result['L2_text'] = L2_text
    result['L2_w2v'] = L2_w2v
    # result['summary_op'] = summary_op
    return result


# TODO
init_learning_rate = FLAGS.learning_rate
# init_learning_rate = 0.1
decay_step = FLAGS.decay_step  # 6000
decay_rate = FLAGS.decay_rate
print('init_learning_rate:{}, decay_step:{}, decay_rate:{}, drop_out:{}'.format(
    init_learning_rate, decay_step, decay_rate, drop_rate))

with tf.device(tf.train.replica_device_setter(worker_device=available_worker_device, cluster=cluster)):
    frame_fea_placeholder = tf.placeholder(dtype=tf.float32,
                                           shape=(None, FRAME_FEAT_LEN, FRAME_FEAT_DIM),
                                           name='frame_feat')
    # tf.placeholder_with_default
    label_1_placeholder = tf.placeholder(dtype=tf.int32,
                                         shape=(None),
                                         name='label_1')
    label_2_placeholder = tf.placeholder(dtype=tf.int32,
                                         shape=(None),
                                         name='label_2')
    title_placeholder = tf.placeholder(dtype=tf.int32,
                                       shape=(None, title_length),
                                       name='title')
    desc_placeholder = tf.placeholder(dtype=tf.int32,
                                      shape=(None, desc_length),
                                      name='desc')
    item_title_placeholder = tf.placeholder(dtype=tf.int32,
                                            shape=(None, item_title_length),
                                            name='item_title')
    item_summary_placeholder = tf.placeholder(dtype=tf.int32,
                                              shape=(None, item_summary_length),
                                              name='item_summary')
    item_cate_placeholder = tf.placeholder(dtype=tf.int32,
                                           shape=(None, item_cate_length),
                                           name='item_cate')
    # mask_placeholder = tf.placeholder(dtype=tf.int32,
    #                                   shape=(None, text_length, text_length),
    #                                   name='mask')

    # frame_in_placeholder = tf.placeholder(dtype=tf.string,
    #                                       shape=(None),
    #                                       name='frame_in')
    word_embed = tf.get_variable('initW', [vocab_size, embed_size], trainable=FLAGS.train_w2v)
    init_embedding = load_embedding(word2vec_table, vocab_size, embed_size, word_embed)

    train_nets = construct_network(frame_fea_placeholder,
                                   label_1=label_1_placeholder,
                                   label_2=label_2_placeholder,
                                   reuse=False, is_training=True,
                                   title_input=title_placeholder,
                                   desc_input=desc_placeholder,
                                   item_title_input=item_title_placeholder,
                                   item_summary_input=item_summary_placeholder,
                                   item_cate_input=item_cate_placeholder,
                                   word_embed=word_embed)
    valid_nets = construct_network(frame_fea_placeholder,
                                   label_1=label_1_placeholder,
                                   label_2=label_2_placeholder,
                                   reuse=True, is_training=False,
                                   title_input=title_placeholder,
                                   desc_input=desc_placeholder,
                                   item_title_input=item_title_placeholder,
                                   item_summary_input=item_summary_placeholder,
                                   item_cate_input=item_cate_placeholder,
                                   word_embed=word_embed)
    # training op
    learning_rate = tf.train.exponential_decay(init_learning_rate, global_step,
                                               decay_step, decay_rate,
                                               staircase=True)
    optimizer = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-08,
                                       use_locking=False)
    train_opt = slim.learning.create_train_op(train_nets['cost'], optimizer, global_step)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)


def get_session(sess):
    session = sess
    while type(session).__name__ != 'Session':
        session = session._sess
    return session


good_model_num = 0
# 3. run session
print("start training")
hooks = [tf.train.StopAtStepHook(last_step=2000000000000000000000)]
step = 0
init_op = [iterator.initializer,
           valid_iterator.initializer,
           tf.global_variables_initializer(),
           tf.local_variables_initializer()]
global_step_init = global_step.assign(0)
saver = tf.train.Saver(max_to_keep=1000)
with tf.train.MonitoredTrainingSession(master=server.target,
                                       checkpoint_dir=FLAGS.buckets,
                                       save_checkpoint_secs=100,
                                       save_summaries_steps=1000,
                                       save_summaries_secs=None,
                                       is_chief=is_chief,
                                       hooks=hooks) as sess:
    # summary_writer = tf.summary.FileWriter(FLAGS.buckets, graph=tf.get_default_graph())
    id_name_map = load_id_name_map('new_struct/train_data_format/video_label_id_name_map_20200609.txt', level_num=3)
    # summary_writer = SummaryWriterCache.get(FLAGS.buckets)
    print('create session done.')
    sess.run(init_op, feed_dict={
        frame_fea_placeholder: np.zeros(shape=(1, FRAME_FEAT_LEN, FRAME_FEAT_DIM), dtype=np.float32),
        label_1_placeholder: np.zeros(shape=(1), dtype=np.int32),
        label_2_placeholder: np.zeros(shape=(1), dtype=np.int32),
        title_placeholder: np.zeros(shape=(1, title_length), dtype=np.int32),
        desc_placeholder: np.zeros(shape=(1, desc_length), dtype=np.int32),
        item_title_placeholder: np.zeros(shape=(1, item_title_length), dtype=np.int32),
        item_summary_placeholder: np.zeros(shape=(1, item_summary_length), dtype=np.int32),
        item_cate_placeholder: np.zeros(shape=(1, item_cate_length), dtype=np.int32)
    })
    sess.run(init_embedding)
    print('initialize word2vec succeed.')
    if FLAGS.restore_name:
        saver.restore(sess, os.path.join(FLAGS.buckets, FLAGS.restore_name))
    sess.run(global_step_init)
    print('restore model succeed.')
    read_time = 0
    pre_process_time_1 = 0
    pre_process_time_2 = 0
    pre_process_time_3 = 0
    train_op_time = 0
    max_accuracy_1 = 0.
    max_accuracy_2 = 0.

    while not sess.should_stop():
        try:
            step += 1
            time_1 = time.time()
            # video_id_batch, frame_fea_batch, label_1_batch, label_2_batch, title_batch, desc_batch, item_title_batch,
            # item_summary_batch, item_cate_batch
            video_id_value, frame_fea_value, label_1_value, label_2_value, title_value, desc_value, \
            item_title_value, item_summary_value, item_cate_value = sess.run(
                [video_id_batch, frame_fea_batch, label_1_batch, label_2_batch, title_batch, desc_batch,
                 item_title_batch, item_summary_batch, item_cate_batch])
            time_2 = time.time()

            label_1_value = np.array(label_1_value)
            label_2_value = np.array(label_2_value)
            frame_feat_new = frame_feat_process(frame_fea_value)

            time_2_1 = time.time()

            title_feat_new = text_feat_process(title_value)
            desc_feat_new = text_feat_process(desc_value)[:, 0:desc_length]
            random_num = random.random()

            item_title_new = text_feat_process_with_drop(item_title_value, drop_rate=FLAGS.drop_item_rate)[:,
                             0:item_title_length]
            item_summary_new = text_feat_process_with_drop(item_summary_value, drop_rate=FLAGS.drop_item_rate)[:,
                               0:item_summary_length]
            item_cate_new = text_feat_process_with_drop(item_cate_value, drop_rate=FLAGS.drop_item_rate)[:,
                            0:item_cate_length]
            # label_weight_value = label_weight_process(root_tags_value, label_weight_map)
            time_2_2 = time.time()

            # text_feat = np.concatenate([title_feat_new, desc_feat_new], axis=1)

            if step == 1:
                print('size of video_id_value:{}'.format(video_id_value.shape))  # (10,)
                print('video ids:{}'.format(video_id_value))
                print('shape of frame_fea_value:{}'.format(frame_feat_new.shape))  # (10, 200, 1024)
                print('shape of label_1_value:{}'.format(label_1_value.shape))  # (10, 1746)
                print('shape of label_2_value:{}'.format(label_2_value.shape))  # (10, 1746)
                print('shape of title_value:{}'.format(title_value.shape))  # (10, )
                print('shape of title_feat_value:{}'.format(title_feat_new.shape))  # (10, 50)
                print('shape of desc_feat_value:{}'.format(desc_feat_new.shape))  # (10, 100)
                print('shape of item_title_new:{}'.format(item_title_new.shape))  # (10, 50)
                print('shape of item_summary_new:{}'.format(item_summary_new.shape))  # (10, 100)
                print('shape of item_cate_new:{}'.format(item_cate_new.shape))  # (10, 50)
                # print('shape of label_weight_value:{}'.format(label_weight_value.shape))  # (10, )

            time_3 = time.time()

            feed_dict = {
                frame_fea_placeholder: frame_feat_new,
                label_1_placeholder: label_1_value,
                label_2_placeholder: label_2_value,
                title_placeholder: title_feat_new,
                desc_placeholder: desc_feat_new,
                item_title_placeholder: item_title_new,
                item_summary_placeholder: item_summary_new,
                item_cate_placeholder: item_cate_new
            }

            learning_rate_val, _, _, global_step_val, loss_root_1, loss_root_2, cost_val, \
            L2_frame_val, L2_text_val, L2_w2v_val = \
                sess.run([learning_rate, train_opt, update_ops, global_step, train_nets['loss_root_1'],
                          train_nets['loss_root_2'],
                          train_nets['cost'], train_nets['L2_frame'], train_nets['L2_text'], train_nets['L2_w2v']],
                         feed_dict=feed_dict)
            # summary_writer.add_summary(summary_str, step)
            time_4 = time.time()
            read_time += time_2 - time_1
            pre_process_time_1 += time_2_1 - time_2
            pre_process_time_2 += time_2_2 - time_2_1
            pre_process_time_3 += time_3 - time_2_2
            train_op_time += time_4 - time_3

            if step <= 10 or step % 100 == 0:
                print("[{}] step:{}, cost:{:.4f}, loss_root_1:{:.4f}, loss_root_2:{:.4f},"
                      "learning_rate:{:.10f}, L2_frame:{:.4f}, L2_text:{:.4f}, L2_w2v:{:.4f}".
                      format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), step, cost_val, loss_root_1, loss_root_2,
                             learning_rate_val, L2_frame_val, L2_text_val, L2_w2v_val))
                if step < 1000:
                    print(
                        'read time:{:.4f}, pre process 1:{:.4f}, pre process 2:{:.4f}, pre process 3:{:.4f}, train time:{:.4f}'.
                            format(read_time, pre_process_time_1, pre_process_time_2, pre_process_time_3,
                                   train_op_time))
        except Exception as e:
            traceback.print_exc()

        # valid
        # if step % 500 == 0 or step == 1:
        # if step == 1 or (step <= 1000 and step % 500 == 0) or (step > 2000 and step % 200 == 0):
        if step == 1 or step % 500 == 0:
            # if step == 1 or (step <= 2000 and step % 200 == 0) or (step > 2000 and step % 200 == 0):
            video_id_all = []

            tags_root_concat = []
            predict_root_concat = []

            gap_root_average_1 = 0
            loss_root_average_1 = 0
            predict_root_all_1 = []
            tags_root_all_1 = []
            root_eval_metrics_1 = EvaluationMetrics(LABEL_1_NUM, 20)

            gap_root_average_2 = 0
            loss_root_average_2 = 0
            predict_root_all_2 = []
            tags_root_all_2 = []
            root_eval_metrics_2 = EvaluationMetrics(LABEL_2_NUM, 20)

            for i in range(valid_batch_num):
                video_id_valid, frame_fea_valid, label_1_valid, label_2_valid, title_fea_valid, desc_fea_valid, \
                item_title_valid, item_summary_valid, item_cate_valid = \
                    sess.run(
                        [valid_video_id, valid_frame_fea, valid_label_1, valid_label_2, valid_title, valid_desc,
                         valid_item_title, valid_item_summary, valid_item_cate])

                label_1_valid = np.array(label_1_valid)
                label_2_valid = np.array(label_2_valid)

                frame_fea_valid = frame_feat_process(frame_fea_valid)

                title_fea_valid = text_feat_process(title_fea_valid)
                desc_fea_valid = text_feat_process(desc_fea_valid)[:, 0:desc_length]

                item_title_valid = text_feat_process(item_title_valid)[:, 0:item_title_length]
                item_summary_valid = text_feat_process(item_summary_valid)[:, 0:item_summary_length]
                item_cate_valid = text_feat_process(item_cate_valid)[:, 0:item_cate_length]

                feed_dict = {
                    frame_fea_placeholder: frame_fea_valid,
                    label_1_placeholder: label_1_valid,
                    label_2_placeholder: label_2_valid,
                    title_placeholder: title_fea_valid,
                    desc_placeholder: desc_fea_valid,

                    item_title_placeholder: item_title_valid,
                    item_summary_placeholder: item_summary_valid,
                    item_cate_placeholder: item_cate_valid
                }

                confidence_root_1, confidence_root_2, loss_root_1, loss_root_2, \
                predict_root_label_1, predict_root_label_2 = \
                    sess.run([valid_nets['confidence_root_1'],
                              valid_nets['confidence_root_2'],
                              valid_nets['loss_root_1'],
                              valid_nets['loss_root_2'],
                              valid_nets['predict_label_root_1'],
                              valid_nets['predict_label_root_2']],
                             feed_dict=feed_dict)

                for idx in range(len(predict_root_label_1)):
                    tags_root_concat.append('{}_{}'.format(int(label_1_valid[idx]), int(label_2_valid[idx])))
                    predict_root_concat.append('{}_{}'.format(int(predict_root_label_1[idx]),
                                                              int(predict_root_label_2[idx])))
                # summary_writer.add_summary(summary_str_valid, step)

                # gap_root = eval_util.calculate_gap(predict_root_label_valid, root_tags_valid)
                # gap_root_average += gap_root
                root_tags_one_hot_1 = one_hot_encode(predict_root_label_1, LABEL_1_NUM)
                root_tags_one_hot_2 = one_hot_encode(predict_root_label_2, LABEL_2_NUM)

                root_eval_metrics_1.accumulate(confidence_root_1, root_tags_one_hot_1,
                                               [0 for i in range(confidence_root_1.shape[0])])
                root_eval_metrics_2.accumulate(confidence_root_2, root_tags_one_hot_2,
                                               [0 for i in range(confidence_root_2.shape[0])])

                video_id_all.extend(video_id_valid)
                predict_root_all_1.append(confidence_root_1)
                predict_root_all_2.append(confidence_root_2)
                tags_root_all_1.append(label_1_valid)
                tags_root_all_2.append(label_2_valid)
                if i == 0:
                    print(confidence_root_1[0, :])
                    print(confidence_root_2[0, :])
                loss_root_average_1 += loss_root_1
                loss_root_average_2 += loss_root_2

            concat_correct_num = 0
            for idx in range(len(tags_root_concat)):
                if tags_root_concat[idx] == predict_root_concat[idx]:
                    concat_correct_num += 1
            concat_correct_rate = concat_correct_num * 1. / len(tags_root_concat)

            tags_root_all_1, predict_root_argmax_1, accuracy_root_1 = \
                cal_accuracy(predict_root_all_1, tags_root_all_1)
            root_matric_result_1 = root_eval_metrics_1.get()

            tags_root_all_2, predict_root_argmax_2, accuracy_root_2 = \
                cal_accuracy(predict_root_all_2, tags_root_all_2)
            root_matric_result_2 = root_eval_metrics_2.get()
            # top_2_acc = top_n_accuracy(np.concatenate(predict_root_all, axis=0), tags_root_all, 2)
            # top_3_acc = top_n_accuracy(np.concatenate(predict_root_all, axis=0), tags_root_all, 3)

            # print('tags root 1:{}'.format(tags_root_all_1[:32]))
            # print('predict root 1:{}'.format(predict_root_argmax_1[:32]))
            # print('tags root 2:{}'.format(tags_root_all_2[:32]))
            # print('predict root 2:{}'.format(predict_root_argmax_2[:32]))
            # print('script file:{}, step:{}, loss_root_1:{:.4f}, acc_root_1:{:.4f}, '
            #       'loss_root_2:{:.4f}, acc_root_2:{:.4f}, '
            #       ' mAP_root_1:{:.4f},gAP_root_1:{:.4f}, mAP_root_2:{:.4f},gAP_root_2:{:.4f}, acc_concat:{:.4f}'.
            #       format(os.path.basename(__file__), step, loss_root_average_1 / valid_batch_num, accuracy_root_1,
            #              loss_root_average_2 / valid_batch_num, accuracy_root_2,
            #              np.mean(root_matric_result_1['aps']), root_matric_result_1['gap'],
            #              np.mean(root_matric_result_2['aps']), root_matric_result_2['gap'],
            #              concat_correct_rate))
            #
            # if step % 2000 == 0:
            #     badcase_set_map = badcase_set_get(tags_root_all_1, predict_root_argmax_1, video_id_all, id_name_map[0])
            #     for pair in badcase_set_map:
            #         if len(badcase_set_map[pair]) > 1:
            #             print('{}:{}'.format(pair, badcase_set_map[pair]))
            #
            # if step > 3000 and accuracy_root_1 > max_accuracy_1 and accuracy_root_2 > max_accuracy_2:
            #     badcase_set_map = badcase_set_get(tags_root_all_1, predict_root_argmax_1, video_id_all, id_name_map[0])
            #     for pair in badcase_set_map:
            #         if len(badcase_set_map[pair]) > 1:
            #             print('{}:{}'.format(pair, badcase_set_map[pair]))
            #
            #     # if accuracy_root >= 0.58:
            #     max_accuracy_1 = accuracy_root_1
            #     max_accuracy_2 = accuracy_root_2
            #     saver.save(get_session(sess), os.path.join(FLAGS.buckets, 'good_model', 'save-{}-{}-{}'.
            #                                                format(step, accuracy_root_1, accuracy_root_2)))
            #     good_model_num += 1
            #     print('save model succeed, step:{}, good_model_num:{}'.format(step, good_model_num))
            #     print_info = True
            #
            #     # summary_writer.close()
