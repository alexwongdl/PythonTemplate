"""
Created by Alex Wang
On 2018-06-27

score:4, count:9
score:5, count:16
score:6, count:45
score:7, count:110
score:8, count:327
score:9, count:574
score:10, count:817
score:11, count:708
score:12, count:387
score:13, count:263
score:14, count:318
score:15, count:325
score:16, count:301
score:17, count:169
score:18, count:30
score:19, count:1

score:6, count:700
score:7, count:880
score:8, count:981
score:9, count:574
score:10, count:817
score:11, count:708
score:12, count:774
score:13, count:789
score:14, count:954
score:15, count:975
score:16, count:903
score:17, count:800
"""
import os
import traceback

import numpy as np
import tensorflow as tf
import cv2

data_root = '/Users/alexwang/data/SCUT-FBP5500_v2'
data_root = '/u02/alexwang/data/beauty/SCUT-FBP5500_v2'
image_dir = os.path.join(data_root, 'Images')
train_dir = os.path.join(data_root, 'train_test_files/5_folders_cross_validations_files/cross_validation_5')

train_file = os.path.join(train_dir, 'train_5.txt')
test_file = os.path.join(train_dir, 'test_5.txt')

train_tfrecords = os.path.join(data_root, 'train_tfrecords_5')
test_tfrecords = os.path.join(data_root, 'test_tfrecords_5')

duplicate_map = {4: 50, 5: 30, 6: 10, 7: 8, 8: 3, 9: 1, 10: 1, 11: 1, 12: 2, 13: 3,
                 14: 3, 15: 3, 16: 3, 17: 4, 18: 10, 19: 50}


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def save_data(label_file, tfrecord_writer, training_data=False):
    with open(label_file, 'r') as reader:
        for line in reader:
            try:
                line = line.strip()
                image_name, label = line.split(' ')
                print(image_name, float(label))
                image_path = os.path.join(image_dir, image_name)
                image = cv2.imread(image_path)
                img_shape = image.shape
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image_rgb = image_rgb.astype(np.float32) / 255.
                score = int(float(label) * 4)
                # if score < 6:
                #     score = 6
                # if score > 17:
                #     score = 17

                img_string = image_rgb.tostring()
                example = tf.train.Example(features=tf.train.Features(feature={
                    'img': bytes_feature(img_string),
                    'label': float_feature(float(label)),
                    'score': int64_feature(score),
                    'width': int64_feature(img_shape[0]),
                    'height': int64_feature(img_shape[1]),
                    'channel': int64_feature(img_shape[2]),
                }))

                if training_data:
                    for i in range(duplicate_map[score]):
                        tfrecord_writer.write(example.SerializeToString())
                else:
                    tfrecord_writer.write(example.SerializeToString())

            except  Exception as e:
                print('error:{}'.format(file))
                traceback.print_exc()


def save_data_to_tfrecords():
    train_tf_writer = tf.python_io.TFRecordWriter(train_tfrecords)
    test_tf_writer = tf.python_io.TFRecordWriter(test_tfrecords)

    save_data(train_file, train_tf_writer, training_data=True)
    save_data(test_file, test_tf_writer)

    train_tf_writer.close()
    test_tf_writer.close()


def read_tfrecords():
    """

    :return:
    """

    score_count = {}

    tfreader = tf.python_io.tf_record_iterator(train_tfrecords)
    i = 0
    for string_record in tfreader:
        example = tf.train.Example()
        example.ParseFromString(string_record)

        img = np.fromstring(example.features.feature['img'].bytes_list.value[0], np.float32)
        width = int(example.features.feature['width'].int64_list.value[0])
        height = int(example.features.feature['height'].int64_list.value[0])
        channel = int(example.features.feature['channel'].int64_list.value[0])
        score = int(example.features.feature['score'].int64_list.value[0])
        img_reshape = np.reshape(img, (width, height, channel))
        label = example.features.feature['label'].float_list.value[0]
        print(len(img))
        print('label:', label)
        print('image shape:{}'.format(img_reshape.shape))

        # score = int(label * 4)
        if score in score_count.keys():
            score_count[score] += 1
        else:
            score_count[score] = 1
        # cv2.imshow('img' + str(i), cv2.cvtColor(img_reshape, cv2.COLOR_RGB2BGR))
        # i += 1
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    for score in score_count.keys():
        print('score:{}, count:{}'.format(score, score_count[score]))


if __name__ == '__main__':
    save_data_to_tfrecords()
    # read_tfrecords()
