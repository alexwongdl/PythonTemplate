"""
Created by Alex Wang
On 2018-06-27
"""

import os
import time
import math

import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorlayer as tl

import inception_v4
import inception_preprocessing

image_size = inception_v4.inception_v4.default_image_size


def build_inception_model(x_input, y_input, reuse, is_training, dropout):
    arg_scope = inception_v4.inception_v4_arg_scope()
    with slim.arg_scope(arg_scope):
        logits, end_points = inception_v4.inception_v4(x_input, is_training=is_training,
                                                       num_classes=1001,
                                                       dropout_keep_prob=dropout,
                                                       reuse=reuse,
                                                       create_aux_logits=False)

    with tf.variable_scope('Beauty', 'BeautyV1', [end_points, y_input], reuse=reuse) as scope:
        # added by Alex Wang for face beauty predict
        regression_conn = slim.fully_connected(end_points['PreLogitsFlatten'], 1, activation_fn=None,
                                               scope='regression_conn', trainable=is_training)
        regression_sigmoid = tf.sigmoid(regression_conn, name='regression_sigmoid')
        regression = tf.scalar_mul(tf.convert_to_tensor(5., dtype=tf.float32), regression_sigmoid)

        ## define cost
        cost_rmse = tf.losses.mean_squared_error(y_input, regression, scope='cost_rmse')
        L2 = 0
        for w in tl.layers.get_variables_with_name('InceptionV4', True, True):
            L2 += tf.contrib.layers.l2_regularizer(0.0001)(w)
        for w in tl.layers.get_variables_with_name('Beauty', True, True):
            L2 += tf.contrib.layers.l2_regularizer(0.0001)(w)

        cost = cost_rmse + L2

        end_points['regression'] = regression
        end_points['cost_rmse'] = cost_rmse
        end_points['L2'] = L2
        end_points['cost'] = cost

    return end_points


def _parse_ucf_features_train(record):
    features = {"img": tf.FixedLenFeature((), tf.string, default_value=''),
                "label": tf.FixedLenFeature((), tf.float32, default_value=0),
                "width": tf.FixedLenFeature((), tf.int64, default_value=0),
                "height": tf.FixedLenFeature((), tf.int64, default_value=0),
                "channel": tf.FixedLenFeature((), tf.int64, default_value=0)}
    parsed_features = tf.parse_single_example(record, features)
    # for key in parsed_features:
    #     print(key, type(parsed_features[key]))

    # print(type(parsed_features['img']))
    img = tf.decode_raw(parsed_features['img'], tf.float32)
    img_reshape = tf.reshape(img, (
        tf.stack([parsed_features['width'], parsed_features['height'], parsed_features['channel']])))
    img_reshape = inception_preprocessing.preprocess_for_train(
        tf.convert_to_tensor(img_reshape, tf.float32), image_size, image_size,
        bbox=None, fast_mode=False, scope='preprocess_train')

    return img_reshape, parsed_features['width'], parsed_features['height'], parsed_features['channel'], \
           parsed_features['label']


def _parse_ucf_features_test(record):
    features = {"img": tf.FixedLenFeature((), tf.string, default_value=''),
                "label": tf.FixedLenFeature((), tf.float32, default_value=0),
                "width": tf.FixedLenFeature((), tf.int64, default_value=0),
                "height": tf.FixedLenFeature((), tf.int64, default_value=0),
                "channel": tf.FixedLenFeature((), tf.int64, default_value=0)}
    parsed_features = tf.parse_single_example(record, features)
    # for key in parsed_features:
    #     print(key, type(parsed_features[key]))

    # print(type(parsed_features['img']))
    img = tf.decode_raw(parsed_features['img'], tf.float32)
    img_reshape = tf.reshape(img, (
        tf.stack([parsed_features['width'], parsed_features['height'], parsed_features['channel']])))
    img_reshape = inception_preprocessing.preprocess_for_eval(
        tf.convert_to_tensor(img_reshape, tf.float32), image_size, image_size,
        central_fraction=1, scope='preprocess_test')

    return img_reshape, parsed_features['width'], parsed_features['height'], parsed_features['channel'], \
           parsed_features['label']


def train_model(FLAGS):
    batch_size = FLAGS.batch_size

    tfrecords_list = [os.path.join(FLAGS.input_dir, 'train_tfrecords_5')]
    dataset = tf.data.TFRecordDataset(tfrecords_list)
    dataset = dataset.map(_parse_ucf_features_train)
    dataset = dataset.shuffle(buffer_size=200)
    dataset = dataset.repeat(-1).batch(batch_size)

    iterator = dataset.make_initializable_iterator()
    next_elem = iterator.get_next()
    img_reshape, img_width, img_height, img_channel, img_label = next_elem

    # valid
    dataset_valid = tf.data.TFRecordDataset([os.path.join(FLAGS.input_dir, 'test_tfrecords_5')])
    dataset_valid = dataset_valid.map(_parse_ucf_features_test).shuffle(buffer_size=200)
    dataset_valid = dataset_valid.repeat(-1).batch(batch_size)
    iterator_valid = dataset_valid.make_initializable_iterator()
    img_reshape_valid, img_width_valid, img_height_valid, img_channel_valid, img_label_valid = \
        iterator_valid.get_next()

    # build model
    x_input = tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size, 3))
    y_input = tf.placeholder(tf.float32, shape=(None))
    end_points_train = build_inception_model(x_input, y_input, reuse=False,
                                             is_training=True, dropout=FLAGS.dropout)

    end_point_test = build_inception_model(x_input, y_input, reuse=True,
                                           is_training=False, dropout=FLAGS.dropout)

    ## TODO: should defined before train_op
    variables = slim.get_variables_to_restore()
    variables_to_restore = [v for v in variables if v.name.split('/')[0] == 'InceptionV4']

    ## train op
    global_step = tf.train.get_or_create_global_step()
    inc_global_step = tf.assign(global_step, global_step + 1)
    learning_rate = tf.train.exponential_decay(FLAGS.learning_rate, global_step,
                                               FLAGS.decay_step, FLAGS.decay_rate,
                                               staircase=True)

    train_op = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-08, use_locking=False).minimize(
        end_points_train['cost'])

    ## summary op
    cost_summary = tf.summary.scalar('cost', end_points_train['cost'])
    learning_rate_summary = tf.summary.scalar('learning_rate', learning_rate)
    cost_rmse_summary = tf.summary.scalar('cost_rmse', end_points_train['cost_rmse'])
    L2_summary = tf.summary.scalar('L2', end_points_train['L2'])
    rmse_valid_summary = tf.summary.scalar('acc_valid', end_point_test['cost_rmse'])

    ## tf.summary.merge_all is deprecated
    # summary_op = tf.summary.merge_all()
    summary_op = tf.summary.merge([cost_summary, learning_rate_summary,
                                   cost_rmse_summary,
                                   L2_summary])

    saver = tf.train.Saver(variables_to_restore)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(iterator.initializer)
        sess.run(iterator_valid.initializer)
        ## tf.train.SummaryWriter is deprecated
        summary_writer = tf.summary.FileWriter(FLAGS.summary_dir, graph=sess.graph)

        if FLAGS.checkpoint is not None:
            saver.restore(sess, FLAGS.checkpoint)

        for step in range(FLAGS.max_iter):
            start_time = time.time()
            fetches = {'train_op': train_op,
                       'global_step': global_step,
                       'inc_global_step': inc_global_step}

            if (step + 1) % FLAGS.print_info_freq == 0 or step == 0:
                fetches['cost'] = end_points_train['cost']
                fetches['cost_rmse'] = end_points_train['cost_rmse']
                fetches['learning_rate'] = learning_rate
                fetches['L2'] = end_points_train['L2']

            if (step + 1) % FLAGS.summary_freq == 0:
                fetches['summary_op'] = summary_op

            img_reshape_val, img_width_val, img_height_val, img_channel_val, img_label_val = \
                sess.run([img_reshape, img_width, img_height, img_channel, img_label])

            result = sess.run(fetches, feed_dict={
                x_input: img_reshape_val,
                y_input: img_label_val})

            if (step + 1) % FLAGS.save_model_freq == 0:
                print("save model")
                if not os.path.exists(FLAGS.save_model_dir):
                    os.mkdir(FLAGS.save_model_dir)
                saver.save(sess, os.path.join(FLAGS.save_model_dir, 'model'), global_step=global_step)

            if (step + 1) % FLAGS.summary_freq == 0:
                summary_writer.add_summary(result['summary_op'], result['global_step'])

            if (step + 1) % FLAGS.print_info_freq == 0 or step == 0:
                epoch = math.ceil(result['global_step'] * 1.0 / FLAGS.print_info_freq)
                rate = FLAGS.batch_size / (time.time() - start_time)
                print("epoch:{}\t, rate:{:.2f} image/sec".format(epoch, rate))
                print("global step:{}".format(result['global_step']))
                print("cost:{:.4f}".format(result['cost']))
                print("cost rmse:{:.4f}".format(result['cost_rmse']))
                print("L2:{:.4f}".format(result['L2']))
                print("learning rate:{:.6f}".format(result['learning_rate']))
                print("")

            if (step + 1) % FLAGS.valid_freq == 0:
                batch_num = int(1100 / batch_size)
                accuracy_average = 0
                for i_valid in range(batch_num):
                    img_reshape_a, img_width_a, img_height_a, img_channel_a, img_label_a = \
                        sess.run([img_reshape_valid, img_width_valid,
                                  img_height_valid, img_channel_valid, img_label_valid])
                    accuracy, summary_str, global_step_val = sess.run(
                        [end_point_test['cost_rmse'], rmse_valid_summary, global_step],
                        feed_dict={
                            x_input: img_reshape_a,
                            y_input: img_label_a
                        })
                    summary_writer.add_summary(summary_str, global_step_val + i_valid)

                    print('valid accuracy:{:.4f}'.format(accuracy))
                    accuracy_average += accuracy
                accuracy_average /= batch_num
                print('valid average accuracy:{:.4f}'.format(accuracy_average))

        summary_writer.close()
