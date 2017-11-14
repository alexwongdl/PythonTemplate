# coding: utf-8
"""
Created by Alex Wang
On 2017-10-24
"""
import warnings
import argparse

def arg_parse_print(FLAGS):
    """
    FLAGS = parser.parse_args()
    :param FLAGS:
    :return:
    """
    warnings.warn("myprint.py is deprecated, please use printutil instead.")
    print('[Configurations]:')
    for name in FLAGS.__dict__.keys():
        value = FLAGS.__dict__[name]
        if type(value) == float:
            print('\t%s: %f'%(name, value))
        elif type(value) == int:
            print('\t%s: %d'%(name, value))
        elif type(value) == str:
            print('\t%s: %s'%(name, value))
        elif type(value) == bool:
            print('\t%s: %s'%(name, value))
        else:
            print('\t%s: %s' % (name, value))

    print('[End of configuration]')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--xx', default='x')
    parser.add_argument('--oo', default='o')
    FLAGS = parser.parse_args()
    arg_parse_print(FLAGS)