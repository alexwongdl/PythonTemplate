"""
Created by Alex Wang
On 2018-07-30

Model: https://github.com/yule-li/CosFace

[Configurations]:
	lfw_pairs: data/pairs.txt
	embedding_size: 1024
	model_def: models.inception_resnet_v1
	save_model: False
	do_flip: False
	image_width: 112
	lfw_dir: dataset/lfw-112x96
	prewhiten: False
	lfw_nrof_folds: 10
	image_height: 112
	lfw_batch_size: 200
	image_size: 224
	fc_bn: True
	model: models/model-20180626-205832.ckpt-60000
	network_type: sphere_network
	lfw_file_ext: jpg
[End of configuration]
"""
import tensorflow as tf
import tensorflow.contrib.slim as slim
import cv2
import numpy as np

import sphere_network
import utils

image_width = 112
image_height = 112
embedding_size = 1024
# face_threshold = 1.49
# face_threshold = 1.54
face_threshold = 0.95
face_combine_threshold = 0.7
save_threshold_min = 0.5
save_threshold_max = 0.7


class CosFace(object):
    """

    """

    def __init__(self, weight_file):
        config = tf.ConfigProto(log_device_placement=False)
        config.gpu_options.allow_growth = True

        self.__graph = tf.Graph()

        with self.__graph.as_default():
            self.__session = tf.Session(config=config, graph=self.__graph)

            self.images_placeholder = tf.placeholder(tf.float32, shape=(
                None, image_height, image_width, 3), name='image')
            self.phase_train_placeholder = tf.placeholder(tf.bool, name='phase_train')

            prelogits = sphere_network.infer(self.images_placeholder, embedding_size)

            prelogits = slim.batch_norm(prelogits,
                                        is_training=self.phase_train_placeholder,
                                        epsilon=1e-5,
                                        scale=True,
                                        scope='softmax_bn')

            self.embeddings = tf.identity(prelogits)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=3)
            saver.restore(self.__session, weight_file)

    def infer(self, images, do_flip=False):
        """

        :param images: utils.py-->load_data
                        rgb format
                        resize to (image_height, image_width, 3)
                        img = img - 127.5
                        img = img / 128.
        :return:
        """
        feed_dict = {self.images_placeholder: images, self.phase_train_placeholder: False}
        feats = self.__session.run(self.embeddings, feed_dict=feed_dict)
        if do_flip:
            images_flip = [np.fliplr(image) for image in images]
            feed_dict_flip = {self.images_placeholder: images_flip, self.phase_train_placeholder: False}
            feats_flip = self.__session.run(self.embeddings, feed_dict=feed_dict_flip)
            feats = np.concatenate((feats, feats_flip), axis=1)
        feats = utils.l2_normalize(feats)
        return feats

    def data_preprocess(self, image):
        """
        :param image: opencv bgr image
        :return:
        """
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = img_rgb.shape[0:2]
        img_new = np.zeros((image_height, image_width, 3), dtype=np.float64)
        ratio = min(image_height * 1.0 / height, image_width * 1.0 / width)
        new_height, new_width = int(height * ratio), int(width * ratio)
        height_offset, width_offset = (image_height - new_height) //2, (image_width - new_width) // 2
        img_rgb = cv2.resize(img_rgb, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        img_rgb = img_rgb.astype(np.float64)
        img_new[height_offset: height_offset + new_height, width_offset: width_offset + new_width, :] = img_rgb

        # img_new = cv2.resize(img_rgb, (image_height, image_width), interpolation=cv2.INTER_CUBIC)
        # img_new = img_new.astype(np.float64)
        img_new -= 127.5
        img_new /= 128.
        return img_new

    def face_dist(self, embedding_one, embedding_two):
        diff = np.subtract(embedding_one, embedding_two)
        dist = np.sum(np.square(diff))
        return dist

    def face_dist_multiple(self, embeddings_one, embeddings_two):
        diff = np.subtract(embeddings_one, embeddings_two)
        dist = np.sum(np.square(diff), 1)
        return dist

    def __del__(self):
        self.__session.close()
