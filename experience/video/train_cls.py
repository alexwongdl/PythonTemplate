"""
Created by Alex Wang on 20200119
"""
import os
from datetime import datetime

import numpy as np
import random
import time
import tensorlayer as tl
import tensorflow as tf
import tensorflow.contrib.slim as slim
from sklearn.metrics import confusion_matrix

import eval_util
from attention_layer import attention_layer, dropout, create_initializer
from eval_util import EvaluationMetrics
from util import cal_accuracy, one_hot_encode, top_n_accuracy
from layer_unit import se_context_gate, moe_model
import traceback
from layer_unit import NeXtVLAD

tf.app.flags.DEFINE_string('tables', '', 'table_list')
tf.app.flags.DEFINE_integer('task_index', -1, 'worker task index')
tf.app.flags.DEFINE_string('ps_hosts', "", "ps hosts")
tf.app.flags.DEFINE_string('worker_hosts', "", "worker hosts")
tf.app.flags.DEFINE_string('job_name', "", "job name:worker or ps")
tf.app.flags.DEFINE_integer("batch_size", 32, "batch size")
tf.app.flags.DEFINE_float("drop_rate", 0.5, "drop_rate")
tf.app.flags.DEFINE_float("learning_rate", 0.0001, "init learning rate")
tf.app.flags.DEFINE_float("frame_weight", 0.0001, "init learning rate")
tf.app.flags.DEFINE_float("text_weight", 0.001, "init learning rate")
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
tf.app.flags.DEFINE_float("use_cg", False, "use context gate or not")

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
TAG_NUM = 128
valid_batch_num = 160
ATTENTION_EMBED_DIM = FLAGS.attention_embed_dim

# text
vocab_size = 196160
title_length = 50
desc_length = 100
ocr_length = 50
cate_length = 20

text_length = 150
num_filter = 256
embed_size = FLAGS.embed_size
_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5

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

config = tf.ConfigProto(intra_op_parallelism_threads=32)
server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=task_index)
if FLAGS.job_name == "ps":
    server.join()

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


word_embed = tf.get_variable('initW', [vocab_size, embed_size], trainable=FLAGS.train_w2v)
init_embedding = load_embedding(word2vec_table, vocab_size, embed_size, word_embed)

# 1. load data
with tf.device(worker_device):
    # filename_queue = tf.train.string_input_producer([FLAGS.tables], num_epochs=100000000)
    dataset = tf.data.TableRecordDataset([train_table],
                                         record_defaults=(np.int64(1), '', -1, '', '', '', ''),
                                         selected_cols="",
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

    # valid dataset
    valid_dataset = tf.data.TableRecordDataset([valid_table],
                                               record_defaults=(np.int64(1), '', -1, '', '', '', ''),
                                               selected_cols="",
                                               # slice_count=worker_count,
                                               # slice_id=task_index,
                                               num_threads=32,
                                               capacity=256)
    valid_dataset = valid_dataset.batch(batch_size)
    valid_dataset = valid_dataset.repeat(100000000)
    valid_iterator = valid_dataset.make_initializable_iterator()

# 2. construct network
available_worker_device = "/job:worker/task:%d" % (task_index)
with tf.device(tf.train.replica_device_setter(worker_device=available_worker_device, cluster=cluster)):
    global_step = tf.Variable(0, name="global_step", trainable=False)


def construct_network(frame_input, root_tags, reuse, is_training, title_input, desc_input,
                      ocr_input, cate_input):
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
            ocr_raw = tf.nn.embedding_lookup(word_embed, ocr_input)
            cate_raw = tf.nn.embedding_lookup(word_embed, cate_input)

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

            rep_cate_2 = txt_conv(cate_raw, ocr_raw, 2, 'conv2_1')
            rep_cate_3 = txt_conv(cate_raw, ocr_raw, 2, 'conv3_1')
            rep_cate_4 = txt_conv(cate_raw, ocr_raw, 2, 'conv4_1')
            rep_cate_5 = txt_conv(cate_raw, ocr_raw, 2, 'conv5_1')

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
                                attention_mask=None,
                                num_attention_heads=4,
                                size_per_head=64,
                                attention_probs_dropout_prob=0.2,
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
                            attention_output = dropout(attention_output, hidden_dropout_prob)
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
                        layer_output = dropout(layer_output, hidden_dropout_prob)
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

        vlad = slim.dropout(vlad, keep_prob=1. - FLAGS.vlad_drop_rate, is_training=is_training, scope="vlad_dropout")

        # SE context gating
        vlad_dim = vlad.get_shape().as_list()[1]
        # print("VLAD dimension", vlad_dim)
        hidden1_weights = tf.get_variable("hidden1_weights",
                                          [vlad_dim, FLAGS.nextvlad_hidden_size],
                                          initializer=slim.variance_scaling_initializer())

        activation = tf.matmul(vlad, hidden1_weights)
        activation = slim.batch_norm(
            activation,
            center=True,
            scale=True,
            is_training=is_training,
            scope="hidden1_bn",
            fused=False)

        # activation = tf.nn.relu(activation)

        gating_weights_1 = tf.get_variable("gating_weights_1",
                                           [FLAGS.nextvlad_hidden_size,
                                            FLAGS.nextvlad_hidden_size // FLAGS.gating_reduction],
                                           initializer=slim.variance_scaling_initializer())

        gates = tf.matmul(activation, gating_weights_1)

        gates = slim.batch_norm(
            gates,
            center=True,
            scale=True,
            is_training=is_training,
            activation_fn=slim.nn.relu,
            scope="gating_bn")

        gating_weights_2 = tf.get_variable("gating_weights_2",
                                           [FLAGS.nextvlad_hidden_size // FLAGS.gating_reduction,
                                            FLAGS.nextvlad_hidden_size],
                                           initializer=slim.variance_scaling_initializer()
                                           )
        gates = tf.matmul(gates, gating_weights_2)

        gates = tf.sigmoid(gates)

        vlad_activation = tf.multiply(activation, gates)
        # vlad_activation = vlad

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
        video_vector = tf.concat([fc_frame, text_logits, attention_final, vlad_activation], axis=1)  # 1280

        video_vector = tf.layers.dropout(video_vector, drop_rate, training=is_training)
        video_vector = tf.nn.relu(video_vector)

        video_vector = tf.layers.dense(video_vector, 512, name='dense_layer_3')
        total_vector = slim.batch_norm(video_vector,
                                       decay=0.9997,
                                       epsilon=0.001,
                                       is_training=is_training)
        tf.check_numerics(video_vector, 'video_vector is inf or nan')
        # -- root predict
        with tf.variable_scope('root_se_cg', reuse=reuse) as scope:
            root_vector = se_context_gate(total_vector, is_training=is_training, se_hidden_size=512)
        predict_root = tf.layers.dense(root_vector, TAG_NUM, name='pred_root')
        predict_root_label = tf.argmax(predict_root, dimension=-1)
        predict_root_confidence = tf.nn.softmax(predict_root, name='conf_root')
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=predict_root, labels=root_tags)
        loss_root = tf.reduce_mean(cross_entropy)


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
           FLAGS.text_weight * L2_text + FLAGS.w2v_weight * L2_w2v
    result = dict()
    result['loss_root'] = loss_root
    result['cost'] = cost
    result['predict_root'] = predict_root
    result['predict_label_root'] = predict_root_label
    result['confidence_root'] = predict_root_confidence
    result['L2_frame'] = L2_frame
    result['L2_text'] = L2_text
    result['L2_w2v'] = L2_w2v
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
    root_tags_placeholder = tf.placeholder(dtype=tf.int32,
                                           shape=(None),
                                           name='tags')
    title_placeholder = tf.placeholder(dtype=tf.int32,
                                       shape=(None, title_length),
                                       name='title')
    desc_placeholder = tf.placeholder(dtype=tf.int32,
                                      shape=(None, desc_length),
                                      name='desc')
    ocr_placeholder = tf.placeholder(dtype=tf.int32,
                                     shape=(None, ocr_length),
                                     name='ocr')
    cate_placeholder = tf.placeholder(dtype=tf.int32,
                                      shape=(None, cate_length),
                                      name='cate')

    train_nets = construct_network(frame_fea_placeholder,
                                   root_tags_placeholder, reuse=False, is_training=True,
                                   title_input=title_placeholder,
                                   desc_input=desc_placeholder,
                                   ocr_input=ocr_placeholder,
                                   cate_input=cate_placeholder)
    valid_nets = construct_network(frame_fea_placeholder,
                                   root_tags_placeholder, reuse=True, is_training=False,
                                   title_input=title_placeholder,
                                   desc_input=desc_placeholder,
                                   ocr_input=ocr_placeholder,
                                   cate_input=cate_placeholder)
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
    print('create session done.')
    sess.run(init_op)
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
    max_accuracy = 0.

    while not sess.should_stop():
        try:
            step += 1
            time_1 = time.time()
            xxx= sess.run([])
            time_2 = time.time()


            frame_feat_new = frame_feat_process(frame_fea_value)
            root_tags_new = np.asarray(root_tags_value)

            time_2_1 = time.time()


            title_feat_new = text_feat_process(title_value)
            desc_feat_new = text_feat_process(desc_value)[:, 0:100]
            random_num = random.random()

            ocr_feat_new = text_feat_process_with_drop(ocr_value, drop_rate=FLAGS.drop_item_rate)[:, 0:ocr_length]
            cate_feat_new = text_feat_process_with_drop(cate_value, drop_rate=FLAGS.drop_item_rate)[:, 0:cate_length]
            # label_weight_value = label_weight_process(root_tags_value, label_weight_map)
            time_2_2 = time.time()

            # text_feat = np.concatenate([title_feat_new, desc_feat_new], axis=1)

            if step == 1:
                print('size of video_id_value:{}'.format(video_id_value.shape))  # (10,)
                print('video ids:{}'.format(video_id_value))
                print('shape of frame_fea_value:{}'.format(frame_feat_new.shape))  # (10, 200, 1024)
                print('shape of root_tags_new:{}'.format(root_tags_new.shape))  # (10, 1746)
                print('shape of title_value:{}'.format(title_value.shape))  # (10, )
                print('shape of title_feat_value:{}'.format(title_feat_new.shape))  # (10, 50)
                print('shape of desc_feat_value:{}'.format(desc_feat_new.shape))  # (10, 100)
                print('shape of ocr_feat_value:{}'.format(ocr_feat_new.shape))  # (10, 100)
                print('shape of cate_feat_new:{}'.format(cate_feat_new.shape))  # (10, 100)
                # print('shape of label_weight_value:{}'.format(label_weight_value.shape))  # (10, )

            tag_exceed = False
            for i in range(len(root_tags_value)):
                label = root_tags_value[i]
                if label < 0 or label >= TAG_NUM:
                    print('label < 0 or label >= TAG_NUM video_id:{}, label:{}'.format(video_id_value[i], label))
                    tag_exceed = True
            if tag_exceed:
                continue
            time_3 = time.time()

            feed_dict = {
                frame_fea_placeholder: frame_feat_new,
                root_tags_placeholder: root_tags_new,
                title_placeholder: title_feat_new,
                desc_placeholder: desc_feat_new,
                ocr_placeholder: ocr_feat_new,
                cate_placeholder: cate_feat_new
            }

            learning_rate_val, _, _, global_step_val, loss_root, cost_val, \
            L2_frame_val, L2_text_val, L2_w2v_val, predict_label_root = \
                sess.run([learning_rate, train_opt, update_ops, global_step, train_nets['loss_root'],
                          train_nets['cost'], train_nets['L2_frame'], train_nets['L2_text'], train_nets['L2_w2v'],
                          train_nets['predict_label_root']],
                         feed_dict=feed_dict)
            time_4 = time.time()
            read_time += time_2 - time_1
            pre_process_time_1 += time_2_1 - time_2
            pre_process_time_2 += time_2_2 - time_2_1
            pre_process_time_3 += time_3 - time_2_2
            train_op_time += time_4 - time_3

            if step <= 10 or step % 100 == 0:
                print("[{}] step:{}, cost:{:.4f}, loss_root:{:.4f}, "
                      "learning_rate:{:.10f}, L2_frame:{:.4f}, L2_text:{:.4f}, L2_w2v:{:.4f}".
                      format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), step, cost_val, loss_root,
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
            gap_root_average = 0
            loss_root_average = 0
            predict_root_all = []
            tags_root_all = []
            video_id_all = []
            root_eval_metrics = EvaluationMetrics(TAG_NUM, 20)

            for i in range(valid_batch_num):
                xxx = sess.run([])
                ...

                feed_dict = {
                    frame_fea_placeholder: frame_fea_valid,
                    root_tags_placeholder: root_tags_valid,
                    title_placeholder: title_fea_valid,
                    desc_placeholder: desc_fea_valid,
                    ocr_placeholder: ocr_fea_valid,
                    cate_placeholder: cate_fea_valid
                }

                confidence_root, loss_root, predict_root_label_valid = \
                    sess.run([valid_nets['confidence_root'],
                              valid_nets['loss_root'],
                              valid_nets['predict_label_root']],
                             feed_dict=feed_dict)

                # gap_root = eval_util.calculate_gap(predict_root_label_valid, root_tags_valid)
                # gap_root_average += gap_root
                root_tags_one_hot = one_hot_encode(root_tags_valid, TAG_NUM)

                root_eval_metrics.accumulate(confidence_root, root_tags_one_hot,
                                             [0 for i in range(confidence_root.shape[0])])

                video_id_all.extend(video_id_valid)
                predict_root_all.append(confidence_root)
                tags_root_all.append(root_tags_valid)
                if i == 0:
                    print(confidence_root[0, :])
                loss_root_average += loss_root

            tags_root_all, predict_root_argmax, accuracy_root = \
                cal_accuracy(predict_root_all, tags_root_all)
            top_2_acc = top_n_accuracy(np.concatenate(predict_root_all, axis=0), tags_root_all, 2)
            top_3_acc = top_n_accuracy(np.concatenate(predict_root_all, axis=0), tags_root_all, 3)
            root_matric_result = root_eval_metrics.get()

            print('tags root:{}'.format(tags_root_all[:32]))
            print('predict root:{}'.format(predict_root_argmax[:32]))
            if step >= 3000 and step % 1000 == 0:
                cm = confusion_matrix(tags_root_all, predict_root_argmax)
                str_cm = ['[{}]'.format(','.join(map(str, row))) for row in cm]
                print(str_cm)

                str_error = ''
                for i in range(len(video_id_all)):
                    if predict_root_argmax[i] != tags_root_all[i]:
                        str_error += '{}:{}->{};'.format(video_id_all[i], tags_root_all[i], predict_root_argmax[i])
                print(str_error)

            # print('equal_arr:{}'.format(equal_arr))
            print('script file:{}, step:{}, loss_root:{:.4f}, acc_root:{:.4f}, '
                  'top_2_acc:{:.4f}, top_3_acc:{:.4f}'
                  ' mAP_root:{:.4f},gAP_root:{:.4f}'.
                  format(os.path.basename(__file__), step, loss_root_average / valid_batch_num,
                         accuracy_root, top_2_acc, top_3_acc,
                         np.mean(root_matric_result['aps']),
                         root_matric_result['gap']))

            if step > 5000 and accuracy_root > max_accuracy:
                # if accuracy_root >= 0.58:
                max_accuracy = accuracy_root
                saver.save(get_session(sess), os.path.join(FLAGS.buckets, 'good_model', 'save-{}-{}'.
                                                           format(step, accuracy_root)))
                good_model_num += 1
                print('save model succeed, step:{}, good_model_num:{}'.format(step, good_model_num))
                print_info = True
