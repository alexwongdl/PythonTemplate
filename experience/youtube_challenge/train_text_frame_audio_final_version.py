"""
Created by Alex Wang on 2018-09-09
training deep learning model with synchronous distributed mode


"""

import os
import numpy as np
import tensorlayer as tl
import tensorflow as tf
import tensorflow.contrib.slim as slim

from data_process import TAG_NUM, FRAME_FEAT_LEN, FRAME_FEAT_DIM, tags_process, frame_feat_process
from data_process import AUDIO_FEAT_LEN, AUDIO_FEAT_DIM, audio_feat_process
from data_process import text_feat_propress, image_feat_process
import eval_util
import traceback

tf.app.flags.DEFINE_string('tables', '', 'table_list')
tf.app.flags.DEFINE_string('task_index', None, 'worker task index')
tf.app.flags.DEFINE_string('ps_hosts', "", "ps hosts")
tf.app.flags.DEFINE_string('worker_hosts', "", "worker hosts")
tf.app.flags.DEFINE_string('job_name', "", "job name:worker or ps")
tf.app.flags.DEFINE_string("buckets", None, "oss buckets")
FLAGS = tf.app.flags.FLAGS

print('tables:', FLAGS.tables)
print('task_index:', FLAGS.task_index)
print('ps_hosts', FLAGS.ps_hosts)
print('worker_hosts', FLAGS.worker_hosts)
print('job_name', FLAGS.job_name)

# parameters
batch_size = 32
# TODO
drop_rate = 0.5
# text
vocab_size = 135000
embed_size = 100
title_length = 50
desc_length = 100
num_filter = 256
_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5

tables = FLAGS.tables.split(",")
train_table = tables[0]
valid_table = tables[1]
print('train_table:{}'.format(train_table))
print('valid_table:{}'.format(valid_table))

ps_spec = FLAGS.ps_hosts.split(",")
worker_spec = FLAGS.worker_hosts.split(",")
cluster = tf.train.ClusterSpec({"ps": ps_spec, "worker": worker_spec})
worker_count = len(worker_spec)
task_index = int(FLAGS.task_index)

is_chief = task_index == 0  # regard worker with index 0 as chief
print('is chief:', is_chief)

server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=task_index)
if FLAGS.job_name == "ps":
    server.join()

worker_device = "/job:worker/task:%d/cpu:%d" % (task_index, 0)
print("worker device:", worker_device)

# 1. load data
with tf.device(worker_device):
    # filename_queue = tf.train.string_input_producer([FLAGS.tables], num_epochs=100000000)
    dataset = tf.data.TableRecordDataset([train_table],
                                         record_defaults=(1, '', '', '', '', '', ''),
                                         selected_cols="vdo_id,vdo_feature,ado_feature,tags,t_encode,d_encode,img_feature",
                                         slice_count=worker_count,
                                         slice_id=task_index)

    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat(100000000)
    dataset = dataset.shuffle(400)
    iterator = dataset.make_initializable_iterator()
    next_elems = iterator.get_next()
    (video_id_batch, frame_fea_batch, audio_fea_batch, tags_batch,
     title_batch, desc_batch, img_batch) = next_elems

    # valid dataset
    valid_dataset = tf.data.TableRecordDataset([valid_table],
                                               record_defaults=(1, '', '', '', '', '', ''),
                                               selected_cols="vdo_id,vdo_feature,ado_feature,tags,t_encode,d_encode,img_feature",
                                               slice_count=worker_count,
                                               slice_id=task_index)
    valid_dataset = valid_dataset.batch(batch_size)
    valid_dataset = valid_dataset.repeat(100000000)
    valid_iterator = valid_dataset.make_initializable_iterator()
    (valid_video_id, valid_frame_fea, valid_audio_fea, valid_tags,
     valid_title_fea, valid_desc_fea, valid_img_fea) = valid_iterator.get_next()

# 2. construct network
available_worker_device = "/job:worker/task:%d" % (task_index)
with tf.device(tf.train.replica_device_setter(worker_device=available_worker_device, cluster=cluster)):
    global_step = tf.Variable(0, name="global_step", trainable=False)


def construct_network(frame_input, x_length, audio_input, tags_input, reuse, is_training,
                      batch_size_in, title_input, desc_input, img_feature):
    """
    :param frame_input:
    :param tags_input:
    :param reuse:
    :param is_training:
    :return:
    """
    with tf.variable_scope('image', reuse=reuse) as scope:
        # [batch_size, 2048]
        image_layer = tf.layers.dense(img_feature, 512, name='dense1')
        image_layer = tf.nn.relu(image_layer)
        image_layer = tf.layers.dense(image_layer, 512, name='dense2')

    with tf.variable_scope('text', reuse=reuse) as scope:
        with tf.device("/cpu:0"), tf.variable_scope('dict'):
            word_embedding = tf.get_variable('initW', [vocab_size, embed_size], trainable=True)
            title_raw = tf.nn.embedding_lookup(word_embedding, title_input)
            desc_raw = tf.nn.embedding_lookup(word_embedding, desc_input)

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

            rep = tf.concat([rep_2, rep_3, rep_4, rep_5], 1)

            text_logits = tf.layers.dense(rep, 512)

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
        nets_frame = tf.nn.relu(nets_frame)
        # max pooling layer
        nets_frame = tf.layers.max_pooling1d(nets_frame, pool_size=4, strides=4, name='pool1d_2')

        # test flat
        nets_frame = tf.layers.flatten(nets_frame)
        # nets_frame = tf.reduce_max(nets_frame, reduction_indices=[1], name='max_pool')

        fc_frame = tf.layers.dense(nets_frame, 512, name='fc1')
        # fc_frame = tf.nn.l2_normalize(fc_frame, dim=1)

    with tf.variable_scope('frame_lstm', reuse=reuse) as scope:
        initializer = tf.random_uniform_initializer(-0.04, 0.04)
        hidden_num = 512

        # layer 1 (batch * 200 * 1024)
        frame_lstm = tf.layers.conv1d(frame_input, filters=1024, kernel_size=3, name='conv1d_1')
        frame_lstm = slim.batch_norm(frame_lstm,
                                     decay=0.9997,
                                     epsilon=0.001,
                                     is_training=is_training)
        frame_lstm = tf.nn.relu(frame_lstm)
        frame_lstm = tf.layers.max_pooling1d(frame_lstm, pool_size=4, strides=4, name='pool1d_1')

        # layer 2 (batch * (198/4-4) * 256) = (batch * 45 * 256)
        frame_lstm = tf.layers.conv1d(frame_lstm, filters=256, kernel_size=5, name='conv1d_2')
        frame_lstm = slim.batch_norm(frame_lstm,
                                     decay=0.9997,
                                     epsilon=0.001,
                                     is_training=is_training)
        frame_lstm = tf.nn.relu(frame_lstm)

        # layer 3 lstm
        frame_lstm = tl.layers.InputLayer(frame_lstm)
        encode_net = tl.layers.DynamicRNNLayer(frame_lstm, cell_fn=tf.nn.rnn_cell.BasicLSTMCell,
                                               cell_init_args={'forget_bias': 0.0, 'state_is_tuple': True},
                                               n_hidden=hidden_num, initializer=initializer, sequence_length=x_length,
                                               return_seq_2d=True, n_layer=2, return_last=True, name='encode_rnn')

        lstm_nets = tf.reshape(encode_net.outputs, (batch_size_in, hidden_num))

    with tf.variable_scope('audio', reuse=reuse) as scope:
        # layer 1  (batch * 400 * 128)
        nets_audio = tf.layers.conv1d(audio_input, filters=2048, kernel_size=5, name='conv1d')
        nets_audio = slim.batch_norm(nets_audio,
                                     decay=0.9997,
                                     epsilon=0.001,
                                     is_training=is_training)
        # global max pooling layer
        nets_audio = tf.reduce_max(nets_audio, reduction_indices=[1], name='max_pool')

        fc_audio = tf.layers.dense(nets_audio, 512, name='fc1')

    with tf.variable_scope('predict', reuse=reuse) as scope:
        video_vector = tf.concat([fc_frame, lstm_nets, fc_audio], axis=1)
        video_vector = tf.layers.dropout(video_vector, drop_rate, training=is_training)
        video_vector = tf.nn.relu(video_vector)
        video_vector = tf.layers.dense(video_vector, 512, name='dense_layer_1')
        total_vector = tf.concat([video_vector, text_logits, image_layer], axis=1)
        total_vector = tf.nn.relu(total_vector)
        total_vector = tf.layers.dense(total_vector, 1024, name='dense_layer_2')
        total_vector = tf.nn.relu(total_vector)
        predict = tf.layers.dense(total_vector, TAG_NUM, name='predict')
        predict_confidence = tf.sigmoid(predict, name='confidence')  # (0,1)
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=predict,
                                                                      labels=tags_input)) * 1000

        L2_frame = 0
        L2_text = 0
        L2_dict = 0
        for w in tl.layers.get_variables_with_name('image', True, True):
            L2_frame += tf.contrib.layers.l2_regularizer(1.0)(w)
        for w in tl.layers.get_variables_with_name('frame', True, True):
            L2_frame += tf.contrib.layers.l2_regularizer(1.0)(w)
        for w in tl.layers.get_variables_with_name('frame_lstm', True, True):
            L2_frame += tf.contrib.layers.l2_regularizer(1.0)(w)
        for w in tl.layers.get_variables_with_name('audio', True, True):
            L2_frame += tf.contrib.layers.l2_regularizer(1.0)(w)
        for w in tl.layers.get_variables_with_name('predict', True, True):
            L2_frame += tf.contrib.layers.l2_regularizer(1.0)(w)

        for w in tl.layers.get_variables_with_name('text/conv', True, True):
            L2_text += tf.contrib.layers.l2_regularizer(1.0)(w)

        for w in tl.layers.get_variables_with_name('text/dict', True, True):
            L2_dict += tf.contrib.layers.l2_regularizer(1.0)(w)

    # TODO : L2_text
    cost = loss + 0.0001 * L2_frame + 0.0001 * L2_text + 0.001 * L2_dict
    result = dict()
    result['loss'] = loss
    result['cost'] = cost
    result['predict'] = predict
    result['confidence'] = predict_confidence
    result['L2_frame'] = L2_frame
    result['L2_text'] = L2_text
    result['L2_dict'] = L2_dict
    return result


# TODO
init_learning_rate = 0.0001
# init_learning_rate = 0.1
decay_step = 3000  # 6000
decay_rate = 0.95
print('init_learning_rate:{}, decay_step:{}, decay_rate:{}, drop_out:{}'.format(
    init_learning_rate, decay_step, decay_rate, drop_rate))

with tf.device(tf.train.replica_device_setter(worker_device=available_worker_device, cluster=cluster)):
    image_fea_placeholder = tf.placeholder(dtype=tf.float32,
                                           shape=(None, 2048),
                                           name='image_feat')

    frame_fea_placeholder = tf.placeholder(dtype=tf.float32,
                                           shape=(None, FRAME_FEAT_LEN, FRAME_FEAT_DIM),
                                           name='frame_feat')
    x_length_placeholder = tf.placeholder(dtype=tf.int32,
                                          shape=(None),
                                          name='x_length')
    audio_fea_placeholder = tf.placeholder(dtype=tf.float32,
                                           shape=(None, AUDIO_FEAT_LEN, AUDIO_FEAT_DIM),
                                           name='audio_feat')
    tags_placeholder = tf.placeholder(dtype=tf.float32,
                                      shape=(None, TAG_NUM),
                                      name='tags')

    title_placeholder = tf.placeholder(dtype=tf.int32,
                                       shape=(None, title_length),
                                       name='title')
    desc_placeholder = tf.placeholder(dtype=tf.int32,
                                      shape=(None, desc_length),
                                      name='desc')

    batch_size_in_placeholder = tf.placeholder(dtype=tf.int32,
                                               shape=(None),
                                               name='batch_size')

    train_nets = construct_network(frame_fea_placeholder, x_length_placeholder,
                                   audio_fea_placeholder,
                                   tags_placeholder, reuse=False, is_training=True,
                                   batch_size_in=batch_size_in_placeholder,
                                   title_input=title_placeholder,
                                   desc_input=desc_placeholder,
                                   img_feature=image_fea_placeholder
                                   )
    valid_nets = construct_network(frame_fea_placeholder, x_length_placeholder,
                                   audio_fea_placeholder,
                                   tags_placeholder, reuse=True, is_training=False,
                                   batch_size_in=batch_size_in_placeholder,
                                   title_input=title_placeholder,
                                   desc_input=desc_placeholder,
                                   img_feature=image_fea_placeholder
                                   )
    # training op
    learning_rate = tf.train.exponential_decay(init_learning_rate, global_step,
                                               decay_step, decay_rate,
                                               staircase=True)
    optimizer = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-08,
                                       use_locking=False)
    train_opt = slim.learning.create_train_op(train_nets['cost'], optimizer, global_step)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

# 3. run session
print("start training")
hooks = [tf.train.StopAtStepHook(last_step=2000000000000000000000)]
step = 0
init_op = [iterator.initializer,
           valid_iterator.initializer,
           tf.global_variables_initializer(),
           tf.local_variables_initializer()]
global_step_init = global_step.assign(0)
saver = tf.train.Saver()
with tf.train.MonitoredTrainingSession(master=server.target,
                                       checkpoint_dir=FLAGS.buckets,
                                       save_checkpoint_secs=100,
                                       save_summaries_steps=1000,
                                       save_summaries_secs=None,
                                       is_chief=is_chief,
                                       hooks=hooks) as sess:
    sess.run(init_op)
    # saver.restore(sess, os.path.join(FLAGS.buckets, 'dist.ckpt-442972'))
    # sess.run(global_step_init)
    # print('restore model succeed.')

    while not sess.should_stop():
        step += 1

        video_id_value, frame_fea_value, audio_fea_value, tags_value, title_value, desc_value, img_value = sess.run(
            [video_id_batch, frame_fea_batch, audio_fea_batch, tags_batch, title_batch, desc_batch, img_batch])

        img_feat_new = image_feat_process(img_value)
        frame_feat_new = frame_feat_process(frame_fea_value)
        x_length = np.ones(dtype=np.int32, shape=(frame_feat_new.shape[0])) * 45
        audio_feat_new = audio_feat_process(audio_fea_value)
        tag_feat_new = tags_process(tags_value)
        title_feat_new = text_feat_propress(title_value)
        desc_feat_new = text_feat_propress(desc_value)
        batch_size_in = frame_feat_new.shape[0]

        if step == 1:
            print('size of video_id_value:{}'.format(video_id_value.shape))  # (10,)
            print('video ids:{}'.format(video_id_value))
            print('shape of frame_fea_value:{}'.format(frame_feat_new.shape))  # (10, 200, 1024)
            print('shape of audio_fea_value:{}'.format(audio_feat_new.shape))  # (10, 400, 128)
            print('shape of title_value:{}'.format(title_value.shape))  # (10, )
            print('shape of title_feat_value:{}'.format(title_feat_new.shape))  # (10, 50)
            print('shape of desc_feat_value:{}'.format(desc_feat_new.shape))  # (10, 100)
            print('shape of tags_value:{}'.format(tag_feat_new.shape))  # (10, 1746)
            print('shape of img_feat_new:{}'.format(img_feat_new.shape))  # (10, 1746)

        if step >= 10000:
            break

        feed_dict = {
            frame_fea_placeholder: frame_feat_new,
            x_length_placeholder: x_length,
            audio_fea_placeholder: audio_feat_new,
            tags_placeholder: tag_feat_new,
            batch_size_in_placeholder: batch_size_in,
            title_placeholder: title_feat_new,
            desc_placeholder: desc_feat_new,
            image_fea_placeholder: img_feat_new
        }
        learning_rate_val, _, _, loss_val, global_step_val, L2_frame, L2_text, L2_dict, cost_val = \
            sess.run(
                [learning_rate, train_opt, update_ops,
                 train_nets['loss'], global_step,
                 train_nets['L2_frame'], train_nets['L2_text'], train_nets['L2_dict'],
                 train_nets['cost']],
                feed_dict=feed_dict)
        if step % 100 == 0 or step < 10:
            print("step:{}, cost:{}, loss:{}, learning_rate:{}, L2_frame:{}, L2_text:{}, L2_dict:{}".
                  format(step, cost_val, loss_val, learning_rate_val, L2_frame, L2_text, L2_dict))

        if step % 1000 == 0:
            print('train_text_frame_audio_44.py')

        # valid
        # if step % 500 == 0 or step == 1:
        if step == 1 or (step <= 2000 and step % 500 == 0) or (step > 2000 and step % 200 == 0):
            score_average = 0
            loss_average = 0
            for i in range(10):
                video_id_valid, frame_fea_valid, audio_fea_valid, tags_valid, title_fea_valid, desc_fea_valid, img_fea_valid = \
                    sess.run(
                        [valid_video_id, valid_frame_fea, valid_audio_fea, valid_tags,
                         valid_title_fea, valid_desc_fea, valid_img_fea])

                image_fea_valid = image_feat_process(img_fea_valid)
                frame_fea_valid = frame_feat_process(frame_fea_valid)
                x_length_valid = np.ones(dtype=np.int32, shape=(frame_fea_valid.shape[0])) * 45
                audio_fea_valid = audio_feat_process(audio_fea_valid)
                title_fea_valid = text_feat_propress(title_fea_valid)
                desc_fea_valid = text_feat_propress(desc_fea_valid)
                tags_valid = tags_process(tags_valid)
                batch_size_in_valid = frame_fea_valid.shape[0]

                feed_dict = {
                    frame_fea_placeholder: frame_fea_valid,
                    x_length_placeholder: x_length_valid,
                    audio_fea_placeholder: audio_fea_valid,
                    tags_placeholder: tags_valid,
                    batch_size_in_placeholder: batch_size_in_valid,
                    title_placeholder: title_fea_valid,
                    desc_placeholder: desc_fea_valid,
                    image_fea_placeholder: image_fea_valid
                }
                predict_result, loss_valid = sess.run([valid_nets['confidence'],
                                                       valid_nets['loss']],
                                                      feed_dict=feed_dict)
                if i == 0:
                    print(predict_result[0, :])

                score = eval_util.calculate_gap(predict_result, tags_valid)
                score_average += score
                loss_average += loss_valid

            print('step:{},  score_average:{:.4f}, loss_average:{:.4f}'.
                  format(step, score_average / 10, loss_average / 10))
