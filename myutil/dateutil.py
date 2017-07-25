"""
Created by Alex Wang
on 2017-07-25
"""

import datetime

def current_date_format():
    """
    获取当前时间字符串
    :return:
    """
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def kdays_ago_date_format(kdays):
    """
    获取kdays天之前的时间字符串
    :param kdays:
    :return:
    """
    kdays_ago_date = datetime.datetime.now() - datetime.timedelta(days=kdays)
    return kdays_ago_date.strftime("%Y-%m-%d %H:%M:%S")


if __name__ == "__main__":
    print(current_date_format())
    print(kdays_ago_date_format(5))

    