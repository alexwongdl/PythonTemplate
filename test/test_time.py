"""
Created by Alex.W.
On 2017-08-09
测试各种时间操作
"""
import time
import datetime

def test_time():
    print('time.time():{}'.format(time.time()))

def test_datetime():
    now = datetime.datetime.now()
    one_min = now + datetime.timedelta(minutes = 1)
    print('now:{}'.format(now))
    print('one min later:{}'.format(one_min))

    time_diff = (one_min - now).seconds
    print('time diff:{}'.format(time_diff))


if __name__ == '__main__':
    test_time()
    test_datetime()

