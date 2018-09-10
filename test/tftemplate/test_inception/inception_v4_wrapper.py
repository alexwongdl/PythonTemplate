"""
Created by Alex Wang
On 20180910

https://github.com/tensorflow/models/tree/master/research/slim
"""
import os

import cv2
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

import inception_v4
import inception_preprocessing
import imagenet

label_names = imagenet.create_readable_names_for_imagenet_labels()


class InceptionV4(object):
    def __init__(self, weight_file):
        config = tf.ConfigProto(log_device_placement=False)
        config.gpu_options.allow_growth = True

        self.__graph = tf.Graph()
        self.image_size = inception_v4.inception_v4.default_image_size

        with self.__graph.as_default():
            self.__session = tf.Session(config=config, graph=self.__graph)

            self.images_placeholder = tf.placeholder(tf.float32, shape=(
                None, self.image_size, self.image_size, 3), name='image')
            self.phase_train_placeholder = tf.placeholder(tf.bool, name='phase_train')

            arg_scope = inception_v4.inception_v4_arg_scope()
            with slim.arg_scope(arg_scope):
                logits, end_points = inception_v4.inception_v4(self.images_placeholder,
                                                               is_training=False,
                                                               num_classes=1001)
            for tensor in tf.all_variables():
                print(tensor.name, tensor.shape)

            self.predict = tf.nn.softmax(logits)
            self.end_points = end_points

            # preprocess
            self.one_image_placeholder = tf.placeholder(tf.float32, shape=(None, None, 3), name='one_image')
            self.preprocessed_image = inception_preprocessing.preprocess_for_eval(
                self.one_image_placeholder, self.image_size, self.image_size,
                central_fraction=1)

            saver = tf.train.Saver(tf.global_variables(), max_to_keep=3)
            saver.restore(self.__session, weight_file)

    def preprocess(self, image):
        """
        :param image: bgr format image
        :return:3-D float numpy image.
        """
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.
        feed_dict = {self.one_image_placeholder: img_rgb}
        image_value = self.__session.run(self.preprocessed_image, feed_dict=feed_dict)
        return image_value

    def infer(self, images):
        """
        :param images:  rgb format images [batch_size, height, width, channel]
        :return:
        """
        feed_dict = {self.images_placeholder: images}
        predict, end_points = self.__session.run([self.predict, self.end_points], feed_dict=feed_dict)

        return predict, end_points


if __name__ == '__main__':
    model_dir = '/Users/alexwang/data/'
    checkpoint_path = os.path.join(model_dir, 'inception_v4.ckpt')

    inception_v4_model = InceptionV4(checkpoint_path)
    img_path = 'laska.png'
    img = cv2.imread(img_path)
    # preprocess
    img = inception_v4_model.preprocess(img)

    imgs = np.expand_dims(img, axis=0)
    predict_value, end_point = inception_v4_model.infer(imgs)
    for predict in predict_value:
        sorted_predict = np.argsort(predict)
        top_5 = sorted_predict[::-1][0:5]
        top_5_labels = [(label_names[i], predict[i]) for i in top_5]
        print(top_5_labels)
