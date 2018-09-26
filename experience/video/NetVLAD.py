"""
Created by Alex Wang on 2018-09-26
training deep learning model with synchronous distributed mode

"""
import os

import numpy as np
import tensorflow as tf
import tensorlayer as tl
import tensorflow.contrib.slim as slim

import eval_util
import data_process
import fangkong_cate_map

tf.app.flags.DEFINE_string('tables', '', 'table_list')
tf.app.flags.DEFINE_string('task_index', None, 'worker task index')
tf.app.flags.DEFINE_string('ps_hosts', "", "ps hosts")
tf.app.flags.DEFINE_string('worker_hosts', "", "worker hosts")
tf.app.flags.DEFINE_string('job_name', "", "job name:worker or ps")
tf.app.flags.DEFINE_string("buckets", None, "oss buckets")

tf.app.flags.DEFINE_float("drop_rate", 0.0, "dropout")
tf.app.flags.DEFINE_float("init_learning_rate", 0.0001, "initialize learning rate")
tf.app.flags.DEFINE_float("decay_step", 6000, "decay steps")
tf.app.flags.DEFINE_float("decay_rate", 0.95, "decay rate")
tf.app.flags.DEFINE_float("cost_weight", 1, "weight of cost")

FLAGS = tf.app.flags.FLAGS

print('tables:', FLAGS.tables)
print('task_index:', FLAGS.task_index)
print('ps_hosts', FLAGS.ps_hosts)
print('worker_hosts', FLAGS.worker_hosts)
print('job_name', FLAGS.job_name)

# define variables
batch_size = 32
FRAME_NUM = 50
FRAME_DIM = 1536

drop_rate = FLAGS.drop_rate
init_learning_rate = FLAGS.init_learning_rate
decay_step = FLAGS.decay_step
decay_rate = FLAGS.decay_rate
cost_weight = FLAGS.cost_weight

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
print(
    'script file:{}, dropout:{}, batch_size:{}, init_learning_rate:{}, decay_step:{}, decay_rate:{}, cost_weight:{}'.format(
        os.path.basename(__file__), drop_rate, batch_size, init_learning_rate, decay_step, decay_rate, cost_weight))

# 1. load data
with tf.device(worker_device):
    # filename_queue = tf.train.string_input_producer([FLAGS.tables], num_epochs=100000000)
    dataset = tf.data.TableRecordDataset([train_table],
                                         record_defaults=(np.int64(1), 1, '', ''),
                                         selected_cols="video_id,frame_num,tag,feature",
                                         slice_count=worker_count,
                                         slice_id=task_index)

    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat(100000000)
    iterator = dataset.make_initializable_iterator()
    next_elems = iterator.get_next()
    (video_id_batch, frame_num_batch, tags_batch, frame_feat_batch) = next_elems

    # valid dataset
    valid_dataset = tf.data.TableRecordDataset([valid_table],
                                               record_defaults=(np.int64(1), 1, '', ''),
                                               selected_cols="video_id,frame_num,tag,feature")
    valid_dataset = valid_dataset.batch(batch_size)
    valid_dataset = valid_dataset.repeat(100000000)
    valid_iterator = valid_dataset.make_initializable_iterator()
    (valid_video_id, valid_frame_num, valid_tags, valid_frame_feat) = valid_iterator.get_next()

# 2. construct network
available_worker_device = "/job:worker/task:%d" % (task_index)
with tf.device(tf.train.replica_device_setter(worker_device=available_worker_device, cluster=cluster)):
    global_step = tf.Variable(0, name="global_step", trainable=False)


def NetVLAD_layer(nets, frame_num, d, word_num=64, scope_name='vlad'):
    """
    implement VLAD layer
    :param nets: (batch_size, frame_num, d)
    :param frame_num:
    :param d:
    :param word_num: words num in visual dictionary
    :param scope_name:
    :return: (batch_size, word_num, d)
    """
    curr_batch_size = tf.shape(nets)[0]
    with tf.variable_scope(scope_name) as scope:
        centers = tf.get_variable(shape=(word_num, d), dtype=nets.dtype, name='centers')  # (word_num, d)

        with tf.variable_scope('similar') as scope:
            # (batch_size, frame_num, word_num)
            distance = tf.layers.conv1d(nets, word_num, kernel_size=1, use_bias=False, name='distance')
            softmax = tf.nn.softmax(distance, dim=-1, name='dist_softmax')

            softmax_expand = tf.expand_dims(softmax, dim=-1)
            softmax_tile = tf.tile(softmax_expand, multiples=[1, 1, 1, d])

        with tf.variable_scope('distance') as scope:
            # (batch_size, frame_num * word_num, d)
            frame_tile = tf.tile(nets, multiples=[1, 1, word_num], name='frame_tile')
            frame_reshape = tf.reshape(frame_tile, shape=(curr_batch_size, frame_num * word_num, d),
                                       name='frame')
            #  (1, word_num, d)
            centers_expand = tf.expand_dims(centers, dim=0)

            # (batch_size, word_num, frame_num * d)
            centers_tile = tf.tile(centers_expand, multiples=[curr_batch_size, frame_num, 1],
                                   name='centers_tile')

            # (batch_size, frame_num * word_num, d)
            substract = frame_reshape - centers_tile
            substract_reshape = tf.reshape(substract, shape=(curr_batch_size, frame_num, word_num, d))

        # reduce_sum
        result = tf.multiply(softmax_tile, substract_reshape)
        result = tf.reduce_sum(result, axis=1)
        return result


def construct_network(frame_feat_input, tags_input, reuse, is_training):
    """
    :param frame_feat_input:(batch, 50, 1536)
    :param tags_input:
    :param reuse:
    :param is_training:
    :return:
    """
    with tf.variable_scope('frame', reuse=reuse) as scope:
        nets = NetVLAD_layer(frame_feat_input, FRAME_NUM, FRAME_DIM)
        nets = slim.batch_norm(nets,
                               decay=0.9997,
                               epsilon=0.001,
                               is_training=is_training)
        nets = tf.layers.conv1d(nets, filters=2048, kernel_size=5, name='conv1d')
        nets = slim.batch_norm(nets,
                               decay=0.9997,
                               epsilon=0.001,
                               is_training=is_training)
        # global max pooling layer
        nets = tf.reduce_max(nets, reduction_indices=[1], name='max_pool')

        fc = tf.layers.dense(nets, 2048, name='fc1')
        fc = tf.layers.dropout(fc, drop_rate, training=is_training)
        fc = tf.nn.relu(fc)

    with tf.variable_scope('predict', reuse=reuse) as scope:
        fc = tf.layers.dense(fc, 516, name='fc2')
        fc = tf.nn.relu(fc)
        predict = tf.layers.dense(fc, fangkong_cate_map.TAG_NUM, name='predict')  # 0-92
        predict_confidence = tf.nn.softmax(predict, name='confidence')  # (0,1)
        cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=predict,
                                                                             labels=tags_input))
        cost = cost * cost_weight

        L2_frame = 0
        L2_predict = 0
        for w in tl.layers.get_variables_with_name('frame', True, True):
            L2_frame += tf.contrib.layers.l2_regularizer(1.0)(w)

        for w in tl.layers.get_variables_with_name('predict', True, True):
            L2_predict += tf.contrib.layers.l2_regularizer(1.0)(w)

        loss = cost + 0.0001 * L2_frame + 0.0001 * L2_predict

        result = dict()
        result['loss'] = loss
        result['cost'] = cost
        result['L2_frame'] = L2_frame
        result['L2_predict'] = L2_predict

        result['predict'] = predict
        result['confidence'] = predict_confidence
        return result


with tf.device(tf.train.replica_device_setter(worker_device=available_worker_device, cluster=cluster)):
    frame_fea_placeholder = tf.placeholder(dtype=tf.float32,
                                           shape=(None, FRAME_NUM, FRAME_DIM),
                                           name='frame_feat')
    tags_placeholder = tf.placeholder(dtype=tf.int32,
                                      shape=(None),
                                      name='tags')

    # def construct_network(frame_feat_input, tags_input, reuse, is_training):
    train_nets = construct_network(frame_fea_placeholder, tags_placeholder,
                                   reuse=False, is_training=True)
    valid_nets = construct_network(frame_fea_placeholder, tags_placeholder,
                                   reuse=True, is_training=False)

    learning_rate = tf.train.exponential_decay(init_learning_rate, global_step,
                                               decay_step, decay_rate,
                                               staircase=True)

    optimizer = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-08,
                                       use_locking=False)
    train_opt = slim.learning.create_train_op(train_nets['loss'], optimizer, global_step)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

# 3. run session
hooks = [tf.train.StopAtStepHook(last_step=2000000000000000000000)]
step = 0
init_op = [iterator.initializer,
           valid_iterator.initializer,
           tf.global_variables_initializer(),
           tf.local_variables_initializer()]

with tf.name_scope('summaries'):
    train_cost = tf.summary.scalar('train_cost', train_nets['cost'])
    valid_cost = tf.summary.scalar('valid_cost', valid_nets['cost'])

    train_merged = tf.summary.merge([train_cost])
    valid_merged = tf.summary.merge([valid_cost])

    train_summary = tf.summary.FileWriter(os.path.join(FLAGS.buckets, 'train_summary'))
    valid_summary = tf.summary.FileWriter(os.path.join(FLAGS.buckets, 'valid_summary'))


def get_session(sess):
    session = sess
    while type(session).__name__ != 'Session':
        session = session._sess
    return session


saver = tf.train.Saver(max_to_keep=1000)
print("start training")
good_model_num = 0
with tf.train.MonitoredTrainingSession(master=server.target,
                                       checkpoint_dir=FLAGS.buckets,
                                       save_checkpoint_secs=100,
                                       save_summaries_steps=None,
                                       save_summaries_secs=None,
                                       is_chief=is_chief,
                                       hooks=hooks) as sess:
    sess.run(init_op)
    while not sess.should_stop():
        step += 1
        # (video_id_batch, frame_num_batch, tags_batch, frame_feat_batch) = next_elems
        video_id_value, frame_num_value, tags_value, frame_feat_value = sess.run(
            [video_id_batch, frame_num_batch, tags_batch, frame_feat_batch])

        frame_feat_value = data_process.frame_process(frame_feat_value)
        tags_value = data_process.tag_process(tags_value)

        if step == 1:
            print('size of video_id_value:{}'.format(video_id_value.shape))  # (b,)
            print('video ids:{}'.format(video_id_value))
            print('shape of frame_fea_value:{}'.format(frame_feat_value.shape))  # (b, 50, 1536)
            print('shape of tags_value:{}'.format(tags_value.shape))  # (b, )
            # print('audio_fea_value:{}'.format(audio_feat_new))

        if step >= 20000:
            break

        feed_dict = {
            frame_fea_placeholder: frame_feat_value,
            tags_placeholder: tags_value
        }
        train_endpoint, _, _, global_step_val, learning_rate_val, train_merged_val = sess.run(
            [train_nets, train_opt, update_ops, global_step, learning_rate, train_merged],
            feed_dict=feed_dict)
        train_summary.add_summary(train_merged_val, step)

        predict_result = train_endpoint['confidence']
        predict_argmax = np.argmax(predict_result, axis=1)
        equal_arr = np.equal(tags_value, predict_argmax)
        acc_num = sum([1 for item in equal_arr if item == True])
        accuracy = acc_num * 1. / len(equal_arr)

        if step % 100 == 0 or step < 10:
            print("step:{}, loss:{}, cost:{}, accuracy:{}, L2_frame:{}, L2_predict:{}, learning_rate:{}".
                  format(step, train_endpoint['loss'], train_endpoint['cost'], accuracy,
                         train_endpoint['L2_frame'], train_endpoint['L2_predict'], learning_rate_val))

        # valid
        if step % 100 == 0 or step == 1:
            cost_average = 0
            valid_batch_num = 10
            predict_all = []
            tags_all = []

            for i in range(valid_batch_num):
                video_id_valid, frame_num_valid, tags_valid, frame_feat_valid = sess.run(
                    [valid_video_id, valid_frame_num, valid_tags, valid_frame_feat])

                frame_feat_valid = data_process.frame_process(frame_feat_valid)
                tags_valid = data_process.tag_process(tags_valid)

                feed_dict = {
                    frame_fea_placeholder: frame_feat_valid,
                    tags_placeholder: tags_valid
                }
                predict_result, cost_valid, valid_merged_val = sess.run(
                    [valid_nets['confidence'], valid_nets['cost'], valid_merged],
                    feed_dict=feed_dict)
                valid_summary.add_summary(valid_merged_val, step + i)

                cost_average += cost_valid

                predict_all.append(predict_result)
                tags_all.append(tags_valid)

            predict_all = np.concatenate(predict_all, axis=0)  # (batch, num_tags)
            tags_all = np.concatenate(tags_all, axis=0)  # (batch,)
            tags_all_one_hot = np.zeros(shape=(len(tags_all), fangkong_cate_map.TAG_NUM))
            tags_all_one_hot[np.arange(len(tags_all)), tags_all] = 1
            score = eval_util.calculate_gap(predict_all, tags_all_one_hot)
            predict_argmax = np.argmax(predict_all, axis=1)

            equal_arr = np.equal(tags_all, predict_argmax)
            acc_num = sum([1 for item in equal_arr if item == True])
            accuracy = acc_num * 1. / len(equal_arr)

            print('tags:{}'.format(tags_all[:32]))
            print('predict:{}'.format(predict_argmax[:32]))
            # print('equal_arr:{}'.format(equal_arr))
            print('script file:{}, step:{}, cost:{:.4f}, GAP:{:.4f}, accuracy:{:.4f}, acc:{}, total:{}'.
                  format(os.path.basename(__file__), step, cost_average / valid_batch_num, score, accuracy,
                         acc_num, len(equal_arr)))
