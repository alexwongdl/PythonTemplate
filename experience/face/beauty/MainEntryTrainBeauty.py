"""
Created by Alex Wang
On 2018-06-11
"""
import os
import argparse
import beauty_model_inception

# os.environ["CUDA_VISIBLE_DEVICES"] = '0'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--task', default='mehod_one', help='method type', type=str)
    """
    common params
    """
    parser.add_argument('--is_training', default=False, help='is training')

    parser.add_argument('--max_iter', default=20000, help='max training iterate times', type=int)
    parser.add_argument('--batch_size', default=16, help='batch size', type=int)
    # learning raate  exponential_decay  decayed_learning_rate = learning_rate * decay_rate ^ (global_step / decay_steps)
    parser.add_argument('--learning_rate', default=0.01, help='learning rate', type=float)
    parser.add_argument('--decay_step', default=10000, help='decay step', type=int)
    parser.add_argument('--decay_rate', default=0.9, help='decay rate', type=float)
    parser.add_argument('--dropout', default=0.9, help='dropout', type=float)

    # model log/save path
    parser.add_argument('--input_dir', default=None, help='input data path', type=str)
    parser.add_argument('--save_model_dir', default=None, help='model dir', type=str)
    parser.add_argument('--save_model_freq', default=10000, help='save check point frequence', type=int)
    parser.add_argument('--summary_dir', default=None, help='summary dir', type=str)
    parser.add_argument('--summary_freq', default=100, help='summary frequency', type=int)
    parser.add_argument('--print_info_freq', default=100, help='print training info frequency', type=int)
    # load checkpoint for initialization or inferencing if checkpoint is not None
    parser.add_argument('--checkpoint', default=None, help='pretrained model', type=str)
    parser.add_argument('--valid_freq', default=1000, help='validate frequence', type=int)

    """
    params for train model
    """
    """
    params for test model
    """
    FLAGS = parser.parse_args()

    if FLAGS.task == 'train_beauty_class':
        print('start train beauty classification model...')
        beauty_model_inception.train_model(FLAGS)
