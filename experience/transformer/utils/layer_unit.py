"""
Created by Alex Wang on 2019-07-10
"""
import math
import tensorflow as tf
import tensorflow.contrib.slim as slim


# from tensorflow import flags

# flags.DEFINE_integer("nextvlad_cluster_size", 64, "Number of units in the NeXtVLAD cluster layer.")
# flags.DEFINE_integer("nextvlad_hidden_size", 1024, "Number of units in the NeXtVLAD hidden layer.")
#
# flags.DEFINE_integer("groups", 8, "number of groups in VLAD encoding")
# flags.DEFINE_float("drop_rate", 0.5, "dropout ratio after VLAD encoding")
# flags.DEFINE_integer("expansion", 2, "expansion ratio in Group NetVlad")
# flags.DEFINE_integer("gating_reduction", 8, "reduction factor in se context gating")
#
# flags.DEFINE_integer("mix_number", 3, "the number of gvlad models")
# flags.DEFINE_float("cl_temperature", 2, "temperature in collaborative learning")
# flags.DEFINE_float("cl_lambda", 1.0, "penalty factor of cl loss")

# Model source: https://github.com/antoine77340/Youtube-8M-WILLOW
class NetVLAD():
    def __init__(self, feature_size, max_frames, cluster_size, add_batch_norm, is_training):
        self.feature_size = feature_size
        self.max_frames = max_frames
        self.is_training = is_training
        self.add_batch_norm = add_batch_norm
        self.cluster_size = cluster_size

    def forward(self, reshaped_input):

        cluster_weights = tf.get_variable("cluster_weights",
                                          [self.feature_size, self.cluster_size],
                                          initializer=tf.random_normal_initializer(
                                              stddev=1 / math.sqrt(self.feature_size)))

        activation = tf.matmul(reshaped_input, cluster_weights, name='activation')

        if self.add_batch_norm:
            activation = slim.batch_norm(
                activation,
                center=True,
                scale=True,
                is_training=self.is_training,
                scope="cluster_bn")
        else:
            cluster_biases = tf.get_variable("cluster_biases",
                                             [self.cluster_size],
                                             initializer=tf.random_normal_initializer(
                                                 stddev=1 / math.sqrt(self.feature_size)))
            activation += cluster_biases

        activation = tf.nn.softmax(activation, name='act_softmax')

        activation = tf.reshape(activation, [-1, self.max_frames, self.cluster_size], name='act_reshape')

        a_sum = tf.reduce_sum(activation, -2, keep_dims=True, name='a_sum')

        cluster_weights2 = tf.get_variable("cluster_weights2",
                                           [1, self.feature_size, self.cluster_size],
                                           initializer=tf.random_normal_initializer(
                                               stddev=1 / math.sqrt(self.feature_size)))

        a = tf.multiply(a_sum, cluster_weights2, name='a')  # broadcast element-wise multiplication

        activation = tf.transpose(activation, perm=[0, 2, 1], name='act_trans')

        reshaped_input = tf.reshape(reshaped_input, [-1, self.max_frames, self.feature_size], name='reshape_input')
        vlad = tf.matmul(activation, reshaped_input, name='vlad')
        vlad = tf.transpose(vlad, perm=[0, 2, 1], name='vlad_trans')
        vlad = tf.subtract(vlad, a, name='vlad_sub')

        vlad = tf.nn.l2_normalize(vlad, 1, name='vlad_norm')

        vlad = tf.reshape(vlad, [-1, self.cluster_size * self.feature_size], name='vlad_reshape')
        vlad = tf.nn.l2_normalize(vlad, 1)

        return vlad


class NeXtVLAD():
    def __init__(self, feature_size, max_frames, cluster_size, is_training=True, expansion=2, groups=None):
        self.feature_size = feature_size
        self.max_frames = max_frames
        self.is_training = is_training
        self.cluster_size = cluster_size
        self.expansion = expansion
        self.groups = groups

    def forward(self, input, mask=None):
        input = slim.fully_connected(input, self.expansion * self.feature_size,
                                     activation_fn=None,
                                     weights_initializer=slim.variance_scaling_initializer())

        attention = slim.fully_connected(input, self.groups, activation_fn=tf.nn.sigmoid,
                                         weights_initializer=slim.variance_scaling_initializer())
        if mask is not None:
            attention = tf.multiply(attention, tf.expand_dims(mask, -1))
        attention = tf.reshape(attention, [-1, self.max_frames * self.groups, 1])
        feature_size = self.expansion * self.feature_size // self.groups

        cluster_weights = tf.get_variable("cluster_weights",
                                          [self.expansion * self.feature_size,
                                           self.groups * self.cluster_size],
                                          initializer=slim.variance_scaling_initializer()
                                          )

        reshaped_input = tf.reshape(input, [-1, self.expansion * self.feature_size])
        activation = tf.matmul(reshaped_input, cluster_weights)

        activation = slim.batch_norm(
            activation,
            center=True,
            scale=True,
            is_training=self.is_training,
            scope="cluster_bn",
            fused=False)

        activation = tf.reshape(activation, [-1, self.max_frames * self.groups, self.cluster_size])
        activation = tf.nn.softmax(activation, dim=-1)
        activation = tf.multiply(activation, attention)
        a_sum = tf.reduce_sum(activation, -2, keep_dims=True)

        cluster_weights2 = tf.get_variable("cluster_weights2",
                                           [1, feature_size, self.cluster_size],
                                           initializer=slim.variance_scaling_initializer()
                                           )
        a = tf.multiply(a_sum, cluster_weights2)

        activation = tf.transpose(activation, perm=[0, 2, 1])

        reshaped_input = tf.reshape(input, [-1, self.max_frames * self.groups, feature_size])
        vlad = tf.matmul(activation, reshaped_input)
        vlad = tf.transpose(vlad, perm=[0, 2, 1])
        vlad = tf.subtract(vlad, a)

        vlad = tf.nn.l2_normalize(vlad, 1)

        # [batch_size, self.cluster_size * feature_size]
        vlad = tf.reshape(vlad, [-1, self.cluster_size * feature_size])
        vlad = slim.batch_norm(vlad,
                               center=True,
                               scale=True,
                               is_training=self.is_training,
                               scope="vlad_bn",
                               fused=False)

        return vlad


class NeXtVLAD_no_bn():
    def __init__(self, feature_size, max_frames, cluster_size, is_training=True, expansion=2, groups=None):
        self.feature_size = feature_size
        self.max_frames = max_frames
        self.is_training = is_training
        self.cluster_size = cluster_size
        self.expansion = expansion
        self.groups = groups

    def forward(self, input, mask=None):
        input = slim.fully_connected(input, self.expansion * self.feature_size,
                                     activation_fn=None,
                                     weights_initializer=slim.variance_scaling_initializer())

        attention = slim.fully_connected(input, self.groups, activation_fn=tf.nn.sigmoid,
                                         weights_initializer=slim.variance_scaling_initializer())
        if mask is not None:
            attention = tf.multiply(attention, tf.expand_dims(mask, -1))
        attention = tf.reshape(attention, [-1, self.max_frames * self.groups, 1])
        feature_size = self.expansion * self.feature_size // self.groups

        cluster_weights = tf.get_variable("cluster_weights",
                                          [self.expansion * self.feature_size,
                                           self.groups * self.cluster_size],
                                          initializer=slim.variance_scaling_initializer()
                                          )

        reshaped_input = tf.reshape(input, [-1, self.expansion * self.feature_size])
        activation = tf.matmul(reshaped_input, cluster_weights)

        # activation = slim.batch_norm(
        #     activation,
        #     center=True,
        #     scale=True,
        #     is_training=self.is_training,
        #     scope="cluster_bn",
        #     fused=False)
        activation = tf.contrib.layers.layer_norm(activation)

        activation = tf.reshape(activation, [-1, self.max_frames * self.groups, self.cluster_size])
        activation = tf.nn.softmax(activation, dim=-1)
        activation = tf.multiply(activation, attention)
        a_sum = tf.reduce_sum(activation, -2, keep_dims=True)

        cluster_weights2 = tf.get_variable("cluster_weights2",
                                           [1, feature_size, self.cluster_size],
                                           initializer=slim.variance_scaling_initializer()
                                           )
        a = tf.multiply(a_sum, cluster_weights2)

        activation = tf.transpose(activation, perm=[0, 2, 1])

        reshaped_input = tf.reshape(input, [-1, self.max_frames * self.groups, feature_size])
        vlad = tf.matmul(activation, reshaped_input)
        vlad = tf.transpose(vlad, perm=[0, 2, 1])
        vlad = tf.subtract(vlad, a)

        vlad = tf.nn.l2_normalize(vlad, 1)

        # [batch_size, self.cluster_size * feature_size]
        vlad = tf.reshape(vlad, [-1, self.cluster_size * feature_size])
        vlad = tf.contrib.layers.layer_norm(vlad)
        # vlad = slim.batch_norm(vlad,
        #                        center=True,
        #                        scale=True,
        #                        is_training=self.is_training,
        #                        scope="vlad_bn",
        #                        fused=False)


        return vlad


def se_context_gate(input, is_training, se_hidden_size=1024):
    """
    :param input: [batch_size, feature_dim]
    :param is_training:
    :param se_hidden_size:
    :return:
    """
    input_dim = input.get_shape().as_list()[1]
    hidden1_weights = tf.get_variable("hidden1_weights",
                                      [input_dim, se_hidden_size],
                                      initializer=slim.variance_scaling_initializer())

    activation = tf.matmul(input, hidden1_weights)
    activation = slim.batch_norm(
        activation,
        center=True,
        scale=True,
        is_training=is_training,
        scope="hidden1_bn",
        fused=False)

    gating_weights_1 = tf.get_variable("gating_weights_1",
                                       [se_hidden_size,
                                        se_hidden_size // 2],
                                       initializer=slim.variance_scaling_initializer())

    gates = tf.matmul(activation, gating_weights_1)

    gates = slim.batch_norm(
        gates,
        center=True,
        scale=True,
        is_training=is_training,
        activation_fn=slim.nn.relu,
        scope="gating_bn")

    gating_weights_2 = tf.get_variable("gating_weights_2",
                                       [se_hidden_size // 2,
                                        se_hidden_size],
                                       initializer=slim.variance_scaling_initializer()
                                       )
    gates = tf.matmul(gates, gating_weights_2)

    gates = tf.sigmoid(gates)

    output = tf.multiply(activation, gates)
    return output


def moe_model(model_input,
              vocab_size,
              num_mixtures=None):
    """Creates a Mixture of (Logistic) Experts model.
    A softmax over a mixture of logistic models.
     The model consists of a per-class softmax distribution over a
     configurable number of logistic classifiers. One of the classifiers in the
     mixture is not trained, and always predicts 0.
    Args:
      model_input: 'batch_size' x 'num_features' matrix of input features.
      vocab_size: The number of classes in the dataset.
      num_mixtures: The number of mixtures (excluding a dummy 'expert' that
        always predicts the non-existence of an entity).
      l2_penalty: How much to penalize the squared magnitudes of parameter
        values.
    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      batch_size x num_classes.
    """
    num_mixtures = num_mixtures

    gate_activations = slim.fully_connected(
        model_input,
        vocab_size * (num_mixtures + 1),
        activation_fn=None,
        biases_initializer=None,
        scope="gates")
    expert_activations = slim.fully_connected(
        model_input,
        vocab_size * num_mixtures,
        activation_fn=None,
        scope="experts")

    gating_distribution = tf.nn.softmax(
        tf.reshape(
            gate_activations,
            [-1, num_mixtures + 1]))  # (Batch * #Labels) x (num_mixtures + 1)
    expert_distribution = tf.nn.sigmoid(
        tf.reshape(expert_activations,
                   [-1, num_mixtures]))  # (Batch * #Labels) x num_mixtures

    final_probabilities_by_class_and_batch = tf.reduce_sum(
        gating_distribution[:, :num_mixtures] * expert_distribution, 1)
    final_probabilities = tf.reshape(final_probabilities_by_class_and_batch,
                                     [-1, vocab_size])
    return final_probabilities
