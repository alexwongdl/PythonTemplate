
import os

import tensorflow as tf
import cv2
import numpy as np
import tensorflow.contrib.slim as slim
import tensorlayer as tl

from attention_layer import attention_layer, dropout, create_initializer
from video_label_data_process import FRAME_FEAT_LEN, FRAME_FEAT_DIM
from layer_unit import NeXtVLAD, se_context_gate

TAG_NUM = 128
# valid_batch_num = 115  # 32*31 = 992--980  32*84=2688--2697  32*115=3680--3689

# text
vocab_size = 196160
title_length = 50
desc_length = 100
ocr_length = 50
cate_length = 20

text_length = 150
num_filter = 256
embed_size = 200
_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5

#  NeXtVLAD
tf.app.flags.DEFINE_integer("nextvlad_cluster_size", 64, "Number of units in the NeXtVLAD cluster layer.")
tf.app.flags.DEFINE_integer("nextvlad_hidden_size", 1024, "Number of units in the NeXtVLAD hidden layer.")

tf.app.flags.DEFINE_integer("groups", 8, "number of groups in VLAD encoding")
tf.app.flags.DEFINE_float("vlad_drop_rate", 0.5, "dropout ratio after VLAD encoding")
tf.app.flags.DEFINE_integer("expansion", 2, "expansion ratio in Group NetVlad")
tf.app.flags.DEFINE_integer("gating_reduction", 8, "reduction factor in se context gating")

# attention
tf.app.flags.DEFINE_integer("attention_embed_dim", 64, "attention embedding dimension")
tf.app.flags.DEFINE_integer("attention_layer_num", 5, "attention layer number")

# text
tf.app.flags.DEFINE_integer("embed_size", 200, "word embedding size")
tf.app.flags.DEFINE_string("restore_name", None, "restore checkpoint")

FLAGS = tf.app.flags.FLAGS
drop_rate = FLAGS.drop_rate
ATTENTION_EMBED_DIM = FLAGS.attention_embed_dim

word_embed = tf.get_variable('initW', [vocab_size, embed_size], trainable=False)


def construct_network(frame_input, root_tags, reuse, is_training, title_input, desc_input, ocr_input, cate_input):
    """
    :param frame_input:
    :param tags_input:
    :param reuse:
    :param is_training:
    :return:
    """
    with tf.variable_scope('text', reuse=reuse) as scope:
        pass

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
        # -- root predict
        with tf.variable_scope('root_se_cg', reuse=reuse) as scope:
            root_vector = se_context_gate(total_vector, is_training=is_training, se_hidden_size=512)
        predict_root = tf.layers.dense(root_vector, TAG_NUM, name='pred_root')
        predict_root_label = tf.argmax(predict_root, dimension=-1)
        predict_root_confidence = tf.nn.softmax(predict_root, name='conf_root')
        loss_root = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits=predict_root, labels=root_tags))

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
    result['hidden_fea'] = root_vector
    return result


def main(_):
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
    # ocr_placeholder = tf.placeholder(dtype=tf.int32,
    #                                  shape=(None, ocr_length),
    #                                  name='desc')
    item_title_placeholder = tf.placeholder(dtype=tf.int32,
                                            shape=(None, ocr_length),
                                            name='item_title')
    cate_placeholder = tf.placeholder(dtype=tf.int32,
                                      shape=(None, cate_length),
                                      name='item_cate')

    valid_nets = construct_network(frame_fea_placeholder,
                                   root_tags_placeholder, reuse=False, is_training=False,
                                   title_input=title_placeholder,
                                   desc_input=desc_placeholder,
                                   ocr_input=item_title_placeholder,
                                   cate_input=cate_placeholder)
    confidence, predict_root_label, hidden_feature = \
        valid_nets['confidence_root'], valid_nets['predict_label_root'], valid_nets['hidden_fea']

    model_dir = '/Users/alexwang/data/'
    model_path = os.path.join(model_dir, 'video_label/models', 'save-6000-0.6193359375')

    saver = tf.train.Saver()

    print [tensor.name for tensor in tf.get_default_graph().as_graph_def().node]
    # print logits
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        tensor_info_frame_fea = tf.saved_model.utils.build_tensor_info(frame_fea_placeholder)
        tensor_info_title = tf.saved_model.utils.build_tensor_info(title_placeholder)
        tensor_info_desc = tf.saved_model.utils.build_tensor_info(desc_placeholder)
        tensor_info_item_title = tf.saved_model.utils.build_tensor_info(item_title_placeholder)
        tensor_info_item_cate = tf.saved_model.utils.build_tensor_info(cate_placeholder)

        tensor_info_confidence = tf.saved_model.utils.build_tensor_info(confidence)
        tensor_info_label = tf.saved_model.utils.build_tensor_info(predict_root_label)
        tensor_info_hidden_fea = tf.saved_model.utils.build_tensor_info(hidden_feature)

        saver.restore(sess, model_path)

        signature = tf.saved_model.signature_def_utils.build_signature_def(
            inputs={'frame_fea': tensor_info_frame_fea,
                    'title': tensor_info_title,
                    'desc': tensor_info_desc,
                    'item_title': tensor_info_item_title,
                    'item_cate': tensor_info_item_cate},
            outputs={'confidence': tensor_info_confidence,
                     'label': tensor_info_label,
                     'hidden_fea': tensor_info_hidden_fea},
            method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME)

        export_dir = os.path.join(model_dir, "saved_model/video_label_v3")
        builder = tf.saved_model.builder.SavedModelBuilder(export_dir)
        builder.add_meta_graph_and_variables(
            sess, [tf.saved_model.tag_constants.SERVING],
            signature_def_map={
                'play_predict': signature
            },
            clear_devices=True)
        builder.save()


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
