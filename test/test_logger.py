"""
Created by Alex Wang on 2018-2-27
测试logger.config
"""

import logging
import logging.config

logging.config.fileConfig('../resource/logger_config.ini')
logger = logging.getLogger('alexwang')

def test_logging_conf():
    logger.info('test logging fileConfig')
    logger.error('an error log')

if __name__ == '__main__':
    test_logging_conf()