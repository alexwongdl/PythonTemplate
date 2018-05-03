"""
Created by Alex Wang on 20170620
"""
from alexutil.logutil import LogUtil
import traceback

logger = LogUtil()

def except_test():
    try:
        fh = open("abc")
    except Exception as e:
        logger.error(e)
        traceback.print_exc()
        logger.error(traceback.format_exc())

def test():
    logger.info("test log")
    except_test()