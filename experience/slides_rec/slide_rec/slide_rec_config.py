import tensorflow as tf

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string("ps_hosts", "", "ps_hosts")
tf.flags.DEFINE_string("worker_hosts", "", "worker_hosts")
tf.flags.DEFINE_string("job_name", "", "job_name")
tf.flags.DEFINE_integer("task_index", "-1", "task_index")
tf.flags.DEFINE_integer("train_steps", 100000000, "max iterations")
tf.flags.DEFINE_integer("batch_size", 16, "minibatch size")
tf.flags.DEFINE_integer("num_epochs", None, "number of epoches")  # None is OK
tf.flags.DEFINE_float("learning_rate", 1e-4, "learning rate")
tf.flags.DEFINE_float("keep_prob", 0.5, "keep_prob")
tf.flags.DEFINE_float("dropout_rate", 0.2, "dropout_rate")
tf.flags.DEFINE_float("l2_weight", 0.0001, "weight of l2 regularization")
tf.flags.DEFINE_integer('decay_step', 6000, 'learning rate decay step')

CROP_SIZE = 224
CHANNEL_NUM = 2
CLIP_LENGTH = 10
INITIAL_LEARNING_RATE = 1e-4
LR_DECAY_FACTOR = 0.5
EPOCHS_PER_LR_DECAY = 2
MOVING_AV_DECAY = 0.9999
MAXWORD_SUMMARY = 110
MAXWORD_TITLE = 18

