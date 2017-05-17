import os

from myutil import stringutil
from test import elegant_code_style
from test.tts import test_gtts
from myutil import configutil

def test_config():
    print(os.path.join(os.getcwd(),"resource\myconfig"))
    path = os.path.join(os.getcwd(),"resource\myconfig")
    print(configutil.get_value(path, "db", "db_user"))
    print(len(configutil.get_value(path, "db", "db_user")))

if __name__ == "__main__":
    print(os.getcwd()) #取的是起始执行目录

    stringutil.test()
    # test_gtts.test()
    test_config()

    elegant_code_style.ifelse(130)
    elegant_code_style.test_cnumerate()
    elegant_code_style.test_zip()
