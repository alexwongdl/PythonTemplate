"""
Created by Alex Wang
On 2018-08-01
"""
import math
import os
import time
import json

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from sklearn import metrics
from scipy.optimize import brentq
from scipy import interpolate
import cv2

import lfw
import cosface_wrapper
import sphere_network
import utils

data_dir = '/Users/alexwang/workspace/CosFace/'
cosface = cosface_wrapper.CosFace(os.path.join(data_dir, 'models/model-20180626-205832.ckpt-60000'))


def test_cosface():
    with open('feature.txt', 'w') as writer:
        # Read the file containing the pairs used for testing
        pairs = lfw.read_pairs(os.path.join(data_dir, 'data/pairs.txt'))
        # pairs = pairs[0:100]
        # pair_len = len(pairs)
        # pairs = np.concatenate((pairs[0:400], pairs[pair_len-401:pair_len-1]))
        # Get the paths for the corresponding images
        paths, actual_issame = lfw.get_paths(os.path.join(data_dir, 'dataset/lfw-112x96'), pairs, 'jpg')

        embedding_size = cosface_wrapper.embedding_size
        # Run forward pass to calculate embeddings
        print('Runnning forward pass on LFW images')
        batch_size = 200
        nrof_images = len(paths)
        nrof_batches = int(math.ceil(1.0 * nrof_images / batch_size))

        emb_array = np.zeros((nrof_images, embedding_size))
        for i in range(nrof_batches):
            start_index = i * batch_size
            print('handing {}/{}'.format(start_index, nrof_images))
            end_index = min((i + 1) * batch_size, nrof_images)
            paths_batch = paths[start_index:end_index]

            # images = utils.load_data(paths_batch, False, False, cosface_wrapper.image_height,
            #                          cosface_wrapper.image_width, False,
            #                          (cosface_wrapper.image_height,
            #                           cosface_wrapper.image_width))
            images = np.zeros((len(paths_batch), cosface_wrapper.image_height,
                               cosface_wrapper.image_width, 3))
            for i in range(len(paths_batch)):
                img = cv2.imread(paths_batch[i])
                images[i, :, :, :] = cosface.data_preprocess(img)

            start_time = time.time()
            feats = cosface.infer(images)
            end_time = time.time()
            cost_time = end_time - start_time
            print('cost time:{}, speed:{} /s'.format(cost_time, batch_size * 1.0 / cost_time))

            emb_array[start_index:end_index, :] = feats
            for write_i in range(len(paths_batch)):
                writer.write('{}\n'.format(json.dumps({'path': paths_batch[write_i],
                                                       'feat': feats[write_i].tolist()})))

        tpr, fpr, accuracy, val, val_std, far = lfw.evaluate(emb_array,
                                                             actual_issame, nrof_folds=10)

        print('Accuracy: %1.3f+-%1.3f' % (np.mean(accuracy), np.std(accuracy)))
        print('Validation rate: %2.5f+-%2.5f @ FAR=%2.5f' % (val, val_std, far))

        auc = metrics.auc(fpr, tpr)
        print('Area Under Curve (AUC): %1.3f' % auc)
        eer = brentq(lambda x: 1. - x - interpolate.interp1d(fpr, tpr)(x), 0., 1.)
        print('Equal Error Rate (EER): %1.3f' % eer)


if __name__ == '__main__':
    test_cosface()
