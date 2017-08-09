"""
Created by Alex.W.
On 2017-08-07
pymysql数据库调用测试
"""

import pymysql
from myutil import configutil
from myutil import dbutil
from myutil import dateutil
import traceback

conf_path = 'E://workspace/dbconf'

# db
online_host = configutil.get_value(conf_path, 'dbonline', 'host')
online_port = configutil.get_value(conf_path, 'dbonline', 'port')
online_user = configutil.get_value(conf_path, 'dbonline', 'user')
online_pass = configutil.get_value(conf_path, 'dbonline', 'pass')

online_conf_dict = dict(host = online_host, port = int(online_port), user = online_user, password = online_pass, db='recsys', charset = 'utf8')

def test_db():
    with dbutil.connect(**online_conf_dict) as conn:
        try:
            with conn.cursor() as cursor:
                sql_str = 'select docid from video_article where insert_time> "{}"'.format(dateutil.kdays_ago_date_format(1))
                print(sql_str)
                cursor.execute(sql_str)
                results = cursor.fetchall()
                print(len(results))
                print(results[0][0])
                print(results[1])

        except Exception as e:
            print(e)
            traceback.print_exc()


if __name__ == '__main__':
    test_db()