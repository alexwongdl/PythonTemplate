# coding: utf-8
'''
Created on 2017-5-25
@author Alex Wang
'''
import logging


class LogUtil:
    def __init__(self, log_path="info.log"):
        logging.basicConfig(level=logging.DEBUG,
                            format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                            datefmt='%a, %d %b %Y %H:%M:%S',
                            filename=log_path,
                            filemode='a')

        # 定义一个StreamHandler，将INFO级别或更高的日志信息打印到标准错误，并将其添加到当前的日志处理对象#
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)

    def debug(self, msg):
        logging.debug(msg)

    def info(self, msg):
        logging.info(msg)

    def error(self, msg):
        logging.error(msg)

    def warning(self, msg):
        logging.warning(msg)


# TODO:所有其他模块使用该logger
logger = LogUtil(log_path='info.log')


def test():
    import logging  # 引入logging模块
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')  # logging.basicConfig函数对日志的输出格式及方式做相关配置
    # 由于日志基本配置中级别设置为DEBUG，所以一下打印信息将会全部显示在控制台上
    logging.info('this is a loggging info message')
    logging.debug('this is a loggging debug message')
    logging.warning('this is loggging a warning message')
    logging.error('this is an loggging error message')
    logging.critical('this is a loggging critical message')
