# coding:utf-8

import os
import traceback

import tensorflow as tf


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

def test_decode_csv():
    word_embed = tf.get_variable('initW', [3, 4], trainable=True)
    name = 'waou'
    # initembedding = load_embedding_local('word2vec_example.txt', 6, 200, word_embed)
    # reader = tf.TextLineReader(skip_header_lines=0)
    # file_queue = tf.train.string_input_producer(['word2vec_example.txt'], name='word_vectors_queue_' + name)
    # _, values = reader.read(file_queue)
    values = tf.constant(['1,2,3,4', '5,6,7,8', '9,10,11,12'])
    embed_raw = tf.decode_csv(
        values, record_defaults=[[''] for _ in range(4)])
    with tf.Session() as sess:
        embed_raw_value = sess.run(embed_raw)
        print(embed_raw_value)


if __name__ == '__main__':
    # word_embed = tf.get_variable('initW', [FLAGS.vocab_size, FLAGS.embed_size], trainable=True)
    # initembedding = load_embedding(embed_table, FLAGS.vocab_size, 100, word_embed)
    # sess.run(initembedding)

    # convert_word2vec_to_odps_format()
    test_decode_csv()
