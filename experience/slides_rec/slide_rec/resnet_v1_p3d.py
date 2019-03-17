# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
Modified to P3D by Alex Wang on 2019-03-17

Contains definitions for the original form of Residual Networks.

The 'v1' residual networks (ResNets) implemented in this module were proposed
by:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385

Other variants were introduced in:
[2] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Identity Mappings in Deep Residual Networks. arXiv: 1603.05027

The networks defined in this module utilize the bottleneck building block of
[1] with projection shortcuts only for increasing depths. They employ batch
normalization *after* every weight layer. This is the architecture used by
MSRA in the Imagenet and MSCOCO 2016 competition models ResNet-101 and
ResNet-152. See [2; Fig. 1a] for a comparison between the current 'v1'
architecture and the alternative 'v2' architecture of [2] which uses batch
normalization *before* every weight layer in the so-called full pre-activation
units.

Typical use:

   from tensorflow.contrib.slim.nets import resnet_v1

ResNet-101 for image classification into 1000 classes:

   # inputs has shape [batch, 224, 224, 3]
   with slim.arg_scope(resnet_v1.resnet_arg_scope()):
      net, end_points = resnet_v1.resnet_v1_101(inputs, 1000, is_training=False)

ResNet-101 for semantic segmentation into 21 classes:

   # inputs has shape [batch, 513, 513, 3]
   with slim.arg_scope(resnet_v1.resnet_arg_scope()):
      net, end_points = resnet_v1.resnet_v1_101(inputs,
                                                21,
                                                is_training=False,
                                                global_pool=False,
                                                output_stride=16)
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import resnet_utils

resnet_arg_scope = resnet_utils.resnet_arg_scope
slim = tf.contrib.slim


class NoOpScope(object):
    """No-op context manager."""

    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc_value, traceback):
        return False


@slim.add_arg_scope
def bottleneck(inputs,
               depth,
               depth_bottleneck,
               stride,
               rate=1,
               outputs_collections=None,
               scope=None,
               use_bounded_activations=False):
    """Bottleneck residual unit variant with BN after convolutions.

    This is the original residual unit proposed in [1]. See Fig. 1(a) of [2] for
    its definition. Note that we use here the bottleneck variant which has an
    extra bottleneck layer.

    When putting together two consecutive ResNet blocks that use this unit, one
    should use stride = 2 in the last unit of the first block.

    Args:
      inputs: A tensor of size [batch, frame, height, width, channels].
      depth: The depth of the ResNet unit output.
      depth_bottleneck: The depth of the bottleneck layers.
      stride: The ResNet unit's stride. Determines the amount of downsampling of
        the units output compared to its input.
      rate: An integer, rate for atrous convolution.
      outputs_collections: Collection to add the ResNet unit output.
      scope: Optional variable_scope.
      use_bounded_activations: Whether or not to use bounded activations. Bounded
        activations better lend themselves to quantized inference.

    Returns:
      The ResNet unit's output.
    """
    with tf.variable_scope(scope, 'bottleneck_v1', [inputs]) as sc:
        depth_in = slim.utils.last_dimension(inputs.get_shape(), min_rank=5)
        if depth == depth_in:
            shortcut = resnet_utils.subsample(inputs, stride, 'shortcut')
        else:
            shortcut = slim.conv3d(
                inputs,
                depth, [1, 1, 1],
                stride=stride,
                activation_fn=tf.nn.relu6 if use_bounded_activations else None,
                scope='shortcut')

        residual = slim.conv3d(inputs, depth_bottleneck, [1, 1, 1], stride=1,
                               scope='conv1')
        residual = resnet_utils.conv3d_same(residual, depth_bottleneck, 3, stride,
                                            rate=rate, scope='conv2')
        residual = slim.conv3d(residual, depth, [1, 1, 1], stride=1,
                               activation_fn=None, scope='conv3')

        if use_bounded_activations:
            # Use clip_by_value to simulate bandpass activation.
            residual = tf.clip_by_value(residual, -6.0, 6.0)
            output = tf.nn.relu6(shortcut + residual)
        else:
            output = tf.nn.relu(shortcut + residual)

        return slim.utils.collect_named_outputs(outputs_collections,
                                                sc.name,
                                                output)


def resnet_v1(inputs,
              blocks,
              num_classes=None,
              is_training=True,
              global_pool=True,
              output_stride=None,
              include_root_block=True,
              spatial_squeeze=True,
              store_non_strided_activations=False,
              dropout_rate=0.2,
              reuse=None,
              scope=None):
    with tf.variable_scope(scope, 'resnet_v1', [inputs], reuse=reuse) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        with slim.arg_scope([slim.conv3d, bottleneck,
                             resnet_utils.stack_blocks_dense],
                            outputs_collections=end_points_collection):
            with (slim.arg_scope([slim.batch_norm], is_training=is_training)
                  if is_training is not None else NoOpScope()):
                net = inputs
                if include_root_block:
                    if output_stride is not None:
                        if output_stride % 4 != 0:
                            raise ValueError('The output_stride needs to be a multiple of 4.')
                        output_stride /= 4
                    # net = resnet_utils.conv3d_same(net, 64, 7, stride=2, scope='conv1')
                    # (n, 4, 224, 224, 2) --> (n, 4, 112, 112, 64)
                    net = slim.conv3d(net, 64, [3, 7, 7], stride=[1, 2, 2], padding='SAME', scope='conv1')
                    # (n, 4, 56, 56, 64)
                    net = slim.max_pool3d(net, [1, 3, 3], stride=[1, 2, 2], scope='pool1')
                    # (n, 4, 28, 28, 64)
                    net = slim.conv3d(net, 64, [3, 7, 7], stride=[1, 2, 2], padding='SAME', scope='conv2')
                # Convert end_points_collection into a dictionary of end_points.
                end_points = slim.utils.convert_collection_to_dict(
                    end_points_collection)

                # (n , 64)
                net = tf.reduce_mean(net, [1, 2, 3], name='pool5', keep_dims=False)
                net = tf.layers.dense(net, 512, name='fc1')
                end_points['embedding512'] = net
                net = tf.layers.dropout(net, rate=dropout_rate, training=is_training, name='dropout')
                net = tf.nn.relu(net)
                net = tf.layers.dense(net, 256, name='fc2')
                end_points['embedding256'] = net
                net = tf.nn.relu(net)
                net = tf.layers.dense(net, units=num_classes, name='full_connect')
                end_points['logits'] = net
                end_points['predictions'] = slim.softmax(net, scope='predictions')
                return net, end_points

def resnet_v7(inputs,
              blocks,
              num_classes=None,
              is_training=True,
              global_pool=True,
              output_stride=None,
              include_root_block=True,
              spatial_squeeze=True,
              store_non_strided_activations=False,
              dropout_rate=0.2,
              reuse=None,
              scope=None):
    with tf.variable_scope(scope, 'resnet_v1', [inputs], reuse=reuse) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        with slim.arg_scope([slim.conv3d, bottleneck,
                             resnet_utils.stack_blocks_dense],
                            outputs_collections=end_points_collection):
            with (slim.arg_scope([slim.batch_norm], is_training=is_training)
                  if is_training is not None else NoOpScope()):
                net = inputs
                if include_root_block:
                    if output_stride is not None:
                        if output_stride % 4 != 0:
                            raise ValueError('The output_stride needs to be a multiple of 4.')
                        output_stride /= 4
                    # net = resnet_utils.conv3d_same(net, 64, 7, stride=2, scope='conv1')
                    # (n, 4, 224, 224, 2) --> (n, 4, 112, 112, 64)
                    conv3d_units_list = FLAGS.conv3d_units.split('|')
                    conv3d_kernels_list = [k.split(',') for k in FLAGS.conv3d_kernels.split('|')]
                    conv3d_strides_list = [s.split(',') for s in FLAGS.conv3d_strides.split('|')]
                    for idx, units, kernels, strides in enumerate(zip(conv3d_units_list,
                                                                      conv3d_kernels,
                                                                      conv3d_strides_list)):
                        # net = slim.conv3d(net, 64, [3, 7, 7], stride=[1, 2, 2], padding='SAME', scope='conv1')
                        net = slim.conv3d(net, units, kernels, stride=strides, padding='SAME', scope='conv1')
                        # (n, 4, 56, 56, 64)
                        if idx != len(conv3d_units_list) - 1:
                            net = slim.max_pool3d(net, [1, 3, 3], stride=[1, 2, 2], scope='pool1')

                # Convert end_points_collection into a dictionary of end_points.
                end_points = slim.utils.convert_collection_to_dict(
                    end_points_collection)

                # (n , 64)
                net = tf.reduce_mean(net, [1, 2, 3], name='pool5', keep_dims=False)
                net = tf.layers.dense(net, 512, name='fc1')
                end_points['embedding512'] = net
                net = tf.layers.dropout(net, rate=dropout_rate, training=is_training, name='dropout')
                net = tf.nn.relu(net)
                net = tf.layers.dense(net, 256, name='fc2')
                end_points['embedding256'] = net
                net = tf.nn.relu(net)
                net = tf.layers.dense(net, units=num_classes, name='full_connect')
                end_points['logits'] = net
                end_points['predictions'] = slim.softmax(net, scope='predictions')
                return net, end_points

def resnet_v6(inputs,
              blocks,
              num_classes=None,
              is_training=True,
              global_pool=True,
              output_stride=None,
              include_root_block=True,
              spatial_squeeze=True,
              store_non_strided_activations=False,
              dropout_rate=0.2,
              reuse=None,
              scope=None):
    with tf.variable_scope(scope, 'resnet_v1', [inputs], reuse=reuse) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        with slim.arg_scope([slim.conv3d, bottleneck,
                             resnet_utils.stack_blocks_dense],
                            outputs_collections=end_points_collection):
            with (slim.arg_scope([slim.batch_norm], is_training=is_training)
                  if is_training is not None else NoOpScope()):
                net = inputs
                if include_root_block:
                    if output_stride is not None:
                        if output_stride % 4 != 0:
                            raise ValueError('The output_stride needs to be a multiple of 4.')
                        output_stride /= 4
                    # net = resnet_utils.conv3d_same(net, 64, 7, stride=2, scope='conv1')
                    # (n, 4, 224, 224, 2) --> (n, 4, 112, 112, 64)
                    net = slim.conv3d(net, 64, [3, 7, 7], stride=[1, 2, 2], padding='SAME', scope='conv1')
                    # (n, 4, 56, 56, 64)
                    net = slim.max_pool3d(net, [1, 3, 3], stride=[1, 2, 2], scope='pool1')
                    # (n, 4, 28, 28, 128)
                    net = slim.conv3d(net, 128, [3, 7, 7], stride=[1, 2, 2], padding='SAME', scope='conv2')
                # net = resnet_utils.stack_blocks_dense(net, blocks, output_stride,
                #                                       store_non_strided_activations)
                # Convert end_points_collection into a dictionary of end_points.
                end_points = slim.utils.convert_collection_to_dict(
                    end_points_collection)

                # (n , 64)
                net = tf.reduce_max(net, [2, 3], name='pool5', keep_dims=False)
                net = tf.layers.conv1d(net, filters=256, kernel_size=3, name='conv1d_1')
                net = slim.batch_norm(net,
                                      decay=0.9997,
                                      epsilon=0.001,
                                      is_training=is_training)
                net = tf.nn.relu(net)
                # (n , 2, 256) --> (n, 512)
                net = tf.layers.flatten(net)
                net = tf.layers.dense(net, 512, name='fc1')
                end_points['embedding512'] = net
                net = tf.layers.dropout(net, rate=dropout_rate, training=is_training, name='dropout')
                net = tf.nn.relu(net)
                net = tf.layers.dense(net, 256, name='fc2')
                end_points['embedding256'] = net
                net = tf.nn.relu(net)
                net = tf.layers.dense(net, units=num_classes, name='full_connect')
                end_points['logits'] = net
                end_points['predictions'] = slim.softmax(net, scope='predictions')
                return net, end_points

def resnet_v2(inputs,
              blocks,
              num_classes=None,
              is_training=True,
              global_pool=True,
              output_stride=None,
              include_root_block=True,
              spatial_squeeze=True,
              store_non_strided_activations=False,
              dropout_rate=0.2,
              reuse=None,
              scope=None):
    with tf.variable_scope(scope, 'resnet_v1', [inputs], reuse=reuse) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        with slim.arg_scope([slim.conv3d, bottleneck,
                             resnet_utils.stack_blocks_dense],
                            outputs_collections=end_points_collection):
            with (slim.arg_scope([slim.batch_norm], is_training=is_training)
                  if is_training is not None else NoOpScope()):
                net = inputs
                if include_root_block:
                    if output_stride is not None:
                        if output_stride % 4 != 0:
                            raise ValueError('The output_stride needs to be a multiple of 4.')
                        output_stride /= 4
                    # net = resnet_utils.conv3d_same(net, 64, 7, stride=2, scope='conv1')
                    net = slim.conv3d(net, 64, [3, 7, 7], stride=[1, 2, 2], padding='SAME', scope='conv1')
                    net = slim.max_pool3d(net, [1, 3, 3], stride=[1, 2, 2], scope='pool1')
                    net = slim.conv3d(net, 128, [3, 7, 7], stride=[1, 2, 2], padding='SAME', scope='conv2')
                # TODO: test different network
                net = slim.conv3d(net, 256, [3, 3, 3], stride=[1, 2, 2], padding='SAME', scope='conv3')
                net = slim.max_pool3d(net, [2, 2, 2], stride=[2, 2, 2], scope='pool2')
                net = slim.conv3d(net, 512, [3, 3, 3], stride=[1, 1, 1], padding='SAME', scope='conv5')
                # net = resnet_utils.stack_blocks_dense(net, blocks, output_stride,
                #                                       store_non_strided_activations)
                # Convert end_points_collection into a dictionary of end_points.
                end_points = slim.utils.convert_collection_to_dict(
                    end_points_collection)

                # modified by Alex Wang on 2018-1-5 (16, 1, 7, 7, 2048) --> (16, 2048)
                net = tf.reduce_max(net, [1, 2, 3], name='pool5', keep_dims=False)
                net = tf.layers.dense(net, 512, name='fc1')
                end_points['embedding512'] = net
                net = tf.layers.dropout(net, rate=dropout_rate, training=is_training, name='dropout')
                net = tf.nn.relu(net)
                net = tf.layers.dense(net, 256, name='fc2')
                end_points['embedding256'] = net
                net = tf.nn.relu(net)
                net = tf.layers.dense(net, units=num_classes, name='full_connect')
                end_points['logits'] = net
                end_points['predictions'] = slim.softmax(net, scope='predictions')
                return net, end_points


def resnet_v3(inputs,
              blocks,
              num_classes=None,
              is_training=True,
              global_pool=True,
              output_stride=None,
              include_root_block=True,
              spatial_squeeze=True,
              store_non_strided_activations=False,
              dropout_rate=0.2,
              reuse=None,
              scope=None):
    with tf.variable_scope(scope, 'resnet_v1', [inputs], reuse=reuse) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        with slim.arg_scope([slim.conv3d, bottleneck,
                             resnet_utils.stack_blocks_dense],
                            outputs_collections=end_points_collection):
            with (slim.arg_scope([slim.batch_norm], is_training=is_training)
                  if is_training is not None else NoOpScope()):
                net = inputs
                if include_root_block:
                    if output_stride is not None:
                        if output_stride % 4 != 0:
                            raise ValueError('The output_stride needs to be a multiple of 4.')
                        output_stride /= 4
                    # net = resnet_utils.conv3d_same(net, 64, 7, stride=2, scope='conv1')
                    net = slim.conv3d(net, 64, [3, 7, 7], stride=[1, 2, 2], padding='SAME', scope='conv1')
                    net = slim.max_pool3d(net, [1, 3, 3], stride=[1, 2, 2], scope='pool1')
                    net = slim.conv3d(net, 64, [3, 7, 7], stride=[1, 2, 2], padding='SAME', scope='conv2')
                    net = slim.conv3d(net, 128, [3, 7, 7], stride=[1, 2, 2], padding='SAME',
                                      scope='conv3')
                # net = resnet_utils.stack_blocks_dense(net, blocks, output_stride,
                #                                       store_non_strided_activations)
                # Convert end_points_collection into a dictionary of end_points.
                end_points = slim.utils.convert_collection_to_dict(
                    end_points_collection)

                # modified by Alex Wang on 2018-1-5 (16, 1, 7, 7, 2048) --> (16, 2048)
                net = tf.reduce_max(net, [1, 2, 3], name='pool5', keep_dims=False)
                net = tf.layers.dense(net, 512, name='fc1')
                end_points['embedding512'] = net
                net = tf.layers.dropout(net, rate=dropout_rate, training=is_training, name='dropout')
                net = tf.nn.relu(net)
                net = tf.layers.dense(net, 256, name='fc2')
                end_points['embedding256'] = net
                net = tf.nn.relu(net)
                net = tf.layers.dense(net, units=num_classes, name='full_connect')
                end_points['logits'] = net
                end_points['predictions'] = slim.softmax(net, scope='predictions')
                return net, end_points


resnet_v1.default_image_size = 224


def resnet_v4(inputs,
              blocks,
              num_classes=None,
              is_training=True,
              global_pool=True,
              output_stride=None,
              include_root_block=True,
              spatial_squeeze=True,
              store_non_strided_activations=False,
              dropout_rate=0.2,
              reuse=None,
              scope=None):
    with tf.variable_scope(scope, 'resnet_v1', [inputs], reuse=reuse) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        with slim.arg_scope([slim.conv3d, bottleneck,
                             resnet_utils.stack_blocks_dense],
                            outputs_collections=end_points_collection):
            with (slim.arg_scope([slim.batch_norm], is_training=is_training)
                  if is_training is not None else NoOpScope()):
                net = inputs
                if include_root_block:
                    if output_stride is not None:
                        if output_stride % 4 != 0:
                            raise ValueError('The output_stride needs to be a multiple of 4.')
                        output_stride /= 4
                    # net = resnet_utils.conv3d_same(net, 64, 7, stride=2, scope='conv1')
                    # (n, 4, 224, 224, 2) --> (n, 4, 112, 112, 64)
                    net = slim.conv3d(net, 64, [3, 7, 7], stride=[1, 2, 2], padding='SAME', scope='conv1')
                    # (n, 4, 56, 56, 64)
                    net = slim.max_pool3d(net, [1, 3, 3], stride=[1, 2, 2], scope='pool1')
                    # (n, 4, 28, 28, 512)
                    net = slim.conv3d(net, 512, [3, 7, 7], stride=[1, 2, 2], padding='SAME', scope='conv2')
                # net = resnet_utils.stack_blocks_dense(net, blocks, output_stride,
                #                                       store_non_strided_activations)
                # Convert end_points_collection into a dictionary of end_points.
                end_points = slim.utils.convert_collection_to_dict(
                    end_points_collection)

                # (n , 4, 512)
                net = tf.reduce_mean(net, [2, 3], name='pool5', keep_dims=False)
                net = tf.layers.conv1d(net, filters=1024, kernel_size=3, name='conv1d_1')
                net = slim.batch_norm(net,
                                      decay=0.9997,
                                      epsilon=0.001,
                                      is_training=is_training)
                net = tf.nn.relu(net)
                # (n , 2, 512) --> (n, 1024)
                net = tf.layers.flatten(net)
                net = tf.layers.dense(net, 512, name='fc1')
                end_points['embedding512'] = net
                net = tf.layers.dropout(net, rate=dropout_rate, training=is_training, name='dropout')
                net = tf.nn.relu(net)
                net = tf.layers.dense(net, 256, name='fc2')
                end_points['embedding256'] = net
                net = tf.nn.relu(net)
                net = tf.layers.dense(net, units=num_classes, name='full_connect')
                end_points['logits'] = net
                end_points['predictions'] = slim.softmax(net, scope='predictions')
                return net, end_points


def resnet_v5(inputs,
              blocks,
              num_classes=None,
              is_training=True,
              global_pool=True,
              output_stride=None,
              include_root_block=True,
              spatial_squeeze=True,
              store_non_strided_activations=False,
              dropout_rate=0.2,
              reuse=None,
              scope=None):
    with tf.variable_scope(scope, 'resnet_v1', [inputs], reuse=reuse) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        with slim.arg_scope([slim.conv3d, bottleneck,
                             resnet_utils.stack_blocks_dense],
                            outputs_collections=end_points_collection):
            with (slim.arg_scope([slim.batch_norm], is_training=is_training)
                  if is_training is not None else NoOpScope()):
                net = inputs
                if include_root_block:
                    if output_stride is not None:
                        if output_stride % 4 != 0:
                            raise ValueError('The output_stride needs to be a multiple of 4.')
                        output_stride /= 4
                    # net = resnet_utils.conv3d_same(net, 64, 7, stride=2, scope='conv1')
                    # (n, 4, 224, 224, 2) --> (n, 4, 112, 112, 64)
                    net = slim.conv3d(net, 64, [3, 7, 7], stride=[1, 2, 2], padding='SAME', scope='conv1')
                    # (n, 4, 56, 56, 64)
                    net = slim.max_pool3d(net, [1, 3, 3], stride=[1, 2, 2], scope='pool1')
                    # (n, 4, 28, 28, 512)
                    net = slim.conv3d(net, 512, [1, 3, 3], stride=[1, 2, 2], padding='SAME', scope='conv2')
                # Convert end_points_collection into a dictionary of end_points.
                end_points = slim.utils.convert_collection_to_dict(
                    end_points_collection)

                # (n , 4, 512)
                net = tf.reduce_max(net, [2, 3], name='pool5', keep_dims=False)
                net = tf.layers.conv1d(net, filters=512, kernel_size=3, name='conv1d_1')
                net = slim.batch_norm(net,
                                      decay=0.9997,
                                      epsilon=0.001,
                                      is_training=is_training)
                net = tf.nn.relu(net)
                net = tf.layers.flatten(net)
                net = tf.layers.dense(net, 512, name='fc1')
                end_points['embedding512'] = net
                net = tf.layers.dropout(net, rate=dropout_rate, training=is_training, name='dropout')
                net = tf.nn.relu(net)
                net = tf.layers.dense(net, 256, name='fc2')
                end_points['embedding256'] = net
                net = tf.nn.relu(net)
                net = tf.layers.dense(net, units=num_classes, name='full_connect')
                end_points['logits'] = net
                end_points['predictions'] = slim.softmax(net, scope='predictions')
                return net, end_points

def resnet_v1_block(scope, base_depth, num_units, stride):
    """Helper function for creating a resnet_v1 bottleneck block.

    Args:
      scope: The scope of the block.
      base_depth: The depth of the bottleneck layer for each unit.
      num_units: The number of units in the block.
      stride: The stride of the block, implemented as a stride in the last unit.
        All other units have stride=1.

    Returns:
      A resnet_v1 bottleneck block.
    """
    return resnet_utils.Block(scope, bottleneck, [{
        'depth': base_depth * 4,
        'depth_bottleneck': base_depth,
        'stride': 1
    }] * (num_units - 1) + [{
        'depth': base_depth * 4,
        'depth_bottleneck': base_depth,
        'stride': stride
    }])


def resnet_v1_50_p3d(inputs,
                     num_classes=None,
                     is_training=True,
                     global_pool=True,
                     output_stride=None,
                     spatial_squeeze=True,
                     store_non_strided_activations=False,
                     reuse=None,
                     scope='resnet_v1_50'):
    """ResNet-50 model of [1]. See resnet_v1() for arg and return description."""
    blocks = [
        resnet_v1_block('block1', base_depth=64, num_units=3, stride=2),
        resnet_v1_block('block2', base_depth=64, num_units=4, stride=2),
        resnet_v1_block('block3', base_depth=128, num_units=6, stride=2),
        resnet_v1_block('block4', base_depth=256, num_units=3, stride=1),
    ]
    return resnet_v1(inputs, blocks, num_classes, is_training,
                     global_pool=global_pool, output_stride=output_stride,
                     include_root_block=True, spatial_squeeze=spatial_squeeze,
                     store_non_strided_activations=store_non_strided_activations,
                     reuse=reuse, scope=scope)


def resnet_v1_50(inputs,
                 num_classes=None,
                 is_training=True,
                 global_pool=True,
                 output_stride=None,
                 spatial_squeeze=True,
                 store_non_strided_activations=False,
                 reuse=None,
                 scope='resnet_v1_50'):
    """ResNet-50 model of [1]. See resnet_v1() for arg and return description."""
    blocks = [
        resnet_v1_block('block1', base_depth=64, num_units=3, stride=2),
        resnet_v1_block('block2', base_depth=128, num_units=4, stride=2),
        resnet_v1_block('block3', base_depth=256, num_units=6, stride=2),
        resnet_v1_block('block4', base_depth=512, num_units=3, stride=1),
    ]
    return resnet_v1(inputs, blocks, num_classes, is_training,
                     global_pool=global_pool, output_stride=output_stride,
                     include_root_block=True, spatial_squeeze=spatial_squeeze,
                     store_non_strided_activations=store_non_strided_activations,
                     reuse=reuse, scope=scope)


def resnet_v1_50_v1(inputs,
                    num_classes=None,
                    is_training=True,
                    global_pool=True,
                    output_stride=None,
                    spatial_squeeze=True,
                    store_non_strided_activations=False,
                    reuse=None,
                    dropout_rate=0.2,
                    scope='resnet_v1_50'):
    """ResNet-50 model of [1]. See resnet_v1() for arg and return description."""

    blocks = [
        resnet_v1_block('block1', base_depth=64, num_units=2, stride=2),
        resnet_v1_block('block2', base_depth=64, num_units=2, stride=2),
        resnet_v1_block('block3', base_depth=256, num_units=2, stride=2)
    ]
    return resnet_v1(inputs, blocks, num_classes, is_training,
                     global_pool=global_pool, output_stride=output_stride,
                     include_root_block=True, spatial_squeeze=spatial_squeeze,
                     store_non_strided_activations=store_non_strided_activations,
                     dropout_rate=dropout_rate,
                     reuse=reuse, scope=scope)

def resnet_v1_50_v7(inputs,
                    num_classes=None,
                    is_training=True,
                    global_pool=True,
                    output_stride=None,
                    spatial_squeeze=True,
                    store_non_strided_activations=False,
                    reuse=None,
                    dropout_rate=0.2,
                    scope='resnet_v1_50'):
    """ResNet-50 model of [1]. See resnet_v1() for arg and return description."""
    blocks = [
        resnet_v1_block('block1', base_depth=64, num_units=2, stride=2),
        resnet_v1_block('block2', base_depth=64, num_units=2, stride=2),
        resnet_v1_block('block3', base_depth=256, num_units=2, stride=2)
    ]
    return resnet_v7(inputs, blocks, num_classes, is_training,
                     global_pool=global_pool, output_stride=output_stride,
                     include_root_block=True, spatial_squeeze=spatial_squeeze,
                     store_non_strided_activations=store_non_strided_activations,
                     dropout_rate=dropout_rate,
                     reuse=reuse, scope=scope)


def resnet_v1_50_v2(inputs,
                    num_classes=None,
                    is_training=True,
                    global_pool=True,
                    output_stride=None,
                    spatial_squeeze=True,
                    store_non_strided_activations=False,
                    reuse=None,
                    dropout_rate=0.2,
                    scope='resnet_v1_50'):
    """ResNet-50 model of [1]. See resnet_v1() for arg and return description."""

    blocks = [
        resnet_v1_block('block1', base_depth=64, num_units=2, stride=2),
        resnet_v1_block('block2', base_depth=64, num_units=2, stride=2),
        resnet_v1_block('block3', base_depth=256, num_units=2, stride=2)
    ]
    return resnet_v2(inputs, blocks, num_classes, is_training,
                     global_pool=global_pool, output_stride=output_stride,
                     include_root_block=True, spatial_squeeze=spatial_squeeze,
                     store_non_strided_activations=store_non_strided_activations,
                     dropout_rate=dropout_rate,
                     reuse=reuse, scope=scope)


def resnet_v1_50_v3(inputs,
                    num_classes=None,
                    is_training=True,
                    global_pool=True,
                    output_stride=None,
                    spatial_squeeze=True,
                    store_non_strided_activations=False,
                    reuse=None,
                    dropout_rate=0.2,
                    scope='resnet_v1_50'):
    """ResNet-50 model of [1]. See resnet_v1() for arg and return description."""
    blocks = [
        resnet_v1_block('block1', base_depth=64, num_units=2, stride=2),
        resnet_v1_block('block2', base_depth=64, num_units=2, stride=2),
        resnet_v1_block('block3', base_depth=256, num_units=2, stride=2)
    ]
    return resnet_v3(inputs, blocks, num_classes, is_training,
                     global_pool=global_pool, output_stride=output_stride,
                     include_root_block=True, spatial_squeeze=spatial_squeeze,
                     store_non_strided_activations=store_non_strided_activations,
                     dropout_rate=dropout_rate,
                     reuse=reuse, scope=scope)

def resnet_v1_50_v4(inputs,
                    num_classes=None,
                    is_training=True,
                    global_pool=True,
                    output_stride=None,
                    spatial_squeeze=True,
                    store_non_strided_activations=False,
                    reuse=None,
                    dropout_rate=0.2,
                    scope='resnet_v1_50'):
    """ResNet-50 model of [1]. See resnet_v1() for arg and return description."""
    blocks = [
        resnet_v1_block('block1', base_depth=64, num_units=2, stride=2),
        resnet_v1_block('block2', base_depth=64, num_units=2, stride=2),
        resnet_v1_block('block3', base_depth=256, num_units=2, stride=2)
    ]
    return resnet_v4(inputs, blocks, num_classes, is_training,
                     global_pool=global_pool, output_stride=output_stride,
                     include_root_block=True, spatial_squeeze=spatial_squeeze,
                     store_non_strided_activations=store_non_strided_activations,
                     dropout_rate=dropout_rate,
                     reuse=reuse, scope=scope)

def resnet_v1_50_v5(inputs,
                    num_classes=None,
                    is_training=True,
                    global_pool=True,
                    output_stride=None,
                    spatial_squeeze=True,
                    store_non_strided_activations=False,
                    reuse=None,
                    dropout_rate=0.2,
                    scope='resnet_v1_50'):
    """ResNet-50 model of [1]. See resnet_v1() for arg and return description."""
    blocks = [
        resnet_v1_block('block1', base_depth=64, num_units=2, stride=2),
        resnet_v1_block('block2', base_depth=64, num_units=2, stride=2),
        resnet_v1_block('block3', base_depth=256, num_units=2, stride=2)
    ]
    return resnet_v5(inputs, blocks, num_classes, is_training,
                     global_pool=global_pool, output_stride=output_stride,
                     include_root_block=True, spatial_squeeze=spatial_squeeze,
                     store_non_strided_activations=store_non_strided_activations,
                     dropout_rate=dropout_rate,
                     reuse=reuse, scope=scope)

def resnet_v1_50_v6(inputs,
                    num_classes=None,
                    is_training=True,
                    global_pool=True,
                    output_stride=None,
                    spatial_squeeze=True,
                    store_non_strided_activations=False,
                    reuse=None,
                    dropout_rate=0.2,
                    scope='resnet_v1_50'):
    """ResNet-50 model of [1]. See resnet_v1() for arg and return description."""

    blocks = [
        resnet_v1_block('block1', base_depth=64, num_units=2, stride=2),
        resnet_v1_block('block2', base_depth=64, num_units=2, stride=2),
        resnet_v1_block('block3', base_depth=256, num_units=2, stride=2)
    ]
    return resnet_v6(inputs, blocks, num_classes, is_training,
                     global_pool=global_pool, output_stride=output_stride,
                     include_root_block=True, spatial_squeeze=spatial_squeeze,
                     store_non_strided_activations=store_non_strided_activations,
                     dropout_rate=dropout_rate,
                     reuse=reuse, scope=scope)


resnet_v1_50.default_image_size = resnet_v1.default_image_size


def resnet_v1_101(inputs,
                  num_classes=None,
                  is_training=True,
                  global_pool=True,
                  output_stride=None,
                  spatial_squeeze=True,
                  store_non_strided_activations=False,
                  reuse=None,
                  scope='resnet_v1_101'):
    """ResNet-101 model of [1]. See resnet_v1() for arg and return description."""
    blocks = [
        resnet_v1_block('block1', base_depth=64, num_units=3, stride=2),
        resnet_v1_block('block2', base_depth=128, num_units=4, stride=2),
        resnet_v1_block('block3', base_depth=256, num_units=23, stride=2),
        resnet_v1_block('block4', base_depth=512, num_units=3, stride=1),
    ]
    return resnet_v1(inputs, blocks, num_classes, is_training,
                     global_pool=global_pool, output_stride=output_stride,
                     include_root_block=True, spatial_squeeze=spatial_squeeze,
                     store_non_strided_activations=store_non_strided_activations,
                     reuse=reuse, scope=scope)


resnet_v1_101.default_image_size = resnet_v1.default_image_size


def resnet_v1_152(inputs,
                  num_classes=None,
                  is_training=True,
                  global_pool=True,
                  output_stride=None,
                  store_non_strided_activations=False,
                  spatial_squeeze=True,
                  reuse=None,
                  scope='resnet_v1_152'):
    """ResNet-152 model of [1]. See resnet_v1() for arg and return description."""
    blocks = [
        resnet_v1_block('block1', base_depth=64, num_units=3, stride=2),
        resnet_v1_block('block2', base_depth=128, num_units=8, stride=2),
        resnet_v1_block('block3', base_depth=256, num_units=36, stride=2),
        resnet_v1_block('block4', base_depth=512, num_units=3, stride=1),
    ]
    return resnet_v1(inputs, blocks, num_classes, is_training,
                     global_pool=global_pool, output_stride=output_stride,
                     include_root_block=True, spatial_squeeze=spatial_squeeze,
                     store_non_strided_activations=store_non_strided_activations,
                     reuse=reuse, scope=scope)


resnet_v1_152.default_image_size = resnet_v1.default_image_size


def resnet_v1_200(inputs,
                  num_classes=None,
                  is_training=True,
                  global_pool=True,
                  output_stride=None,
                  store_non_strided_activations=False,
                  spatial_squeeze=True,
                  reuse=None,
                  scope='resnet_v1_200'):
    """ResNet-200 model of [2]. See resnet_v1() for arg and return description."""
    blocks = [
        resnet_v1_block('block1', base_depth=64, num_units=3, stride=2),
        resnet_v1_block('block2', base_depth=128, num_units=24, stride=2),
        resnet_v1_block('block3', base_depth=256, num_units=36, stride=2),
        resnet_v1_block('block4', base_depth=512, num_units=3, stride=1),
    ]
    return resnet_v1(inputs, blocks, num_classes, is_training,
                     global_pool=global_pool, output_stride=output_stride,
                     include_root_block=True, spatial_squeeze=spatial_squeeze,
                     store_non_strided_activations=store_non_strided_activations,
                     reuse=reuse, scope=scope)


resnet_v1_200.default_image_size = resnet_v1.default_image_size
