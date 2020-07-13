"""
Create by Alex Wang on 2020-05-25
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import os

"""
Estimator模型代码中不需要知道ps_hosts、worker_hosts以及job_name等信息，Estimator会从环境变量TF_CONFIG中获取cluster信息；这里使用task_index和worker_hosts仅是为了数据分片读取需要，当然这些信息可以从TF_CONFIG获得。

如果需要在模型训练时，对模型进行评估，需要在cluster中配置evaluator:-Dcluster="{"worker":{"count":3}, "ps":{"count":2},"evaluator":{"count":1}}"
"""
flags = tf.app.flags
flags.DEFINE_string("tables", "", "tables info")
flags.DEFINE_integer("task_index", None, "Worker or server index")
flags.DEFINE_string("worker_hosts", "", "worker hosts")
FLAGS = tf.app.flags.FLAGS


def model_fn(features, labels, mode):
    W = tf.Variable(tf.zeros([3, 2]))
    b = tf.Variable(tf.zeros([2]))
    pred = tf.matmul(features, W) + b

    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=labels))
    if mode == tf.estimator.ModeKeys.TRAIN:
        global_step = tf.train.get_or_create_global_step()
        opt = tf.train.GradientDescentOptimizer(0.05)
        train_op = opt.minimize(loss, global_step=global_step, name='train_op')
        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
            train_op=train_op)
    elif mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss)
    else:
        raise ValueError("Error mode not supported: {}" % (mode))


def decode_line(line):
    v1, v2, v3, v4 = tf.decode_csv(line, record_defaults=[[1.0]] * 4, field_delim=',')
    labels = tf.cast(v4, tf.int32)
    features = tf.stack([v1, v2, v3])
    return features, labels


def train_input_fn():
    worker_spec = FLAGS.worker_hosts.split(",")
    worker_count = len(worker_spec)
    task_index = FLAGS.task_index

    dataset = tf.data.TableRecordDataset([FLAGS.tables], record_defaults=[""],
                                         slice_id=task_index,
                                         slice_count=worker_count)
    d = dataset.cache().map(decode_line).shuffle(True).batch(128).repeat()
    return d


def eval_input_fn():
    dataset = tf.data.TableRecordDataset([FLAGS.tables], record_defaults=[""])
    d = dataset.cache().map(decode_line).batch(128)
    return d


def main():
    strategy = tf.contrib.distribute.ParameterServerStrategy()
    sess_config = tf.ConfigProto(allow_soft_placement=True)
    config = tf.estimator.RunConfig(train_distribute=strategy,
                                    session_config=sess_config,
                                    save_checkpoints_steps=100,
                                    save_summary_steps=100)
    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        config=config,
        model_dir=FLAGS.checkpointDir)
    train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=10000)
    eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn, start_delay_secs=6, throttle_secs=1)
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    env_dist = os.environ
    print(env_dist.get('TF_CONFIG'))
    tf.app.run()
