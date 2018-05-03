"""
Created by hzwangjian1
On 2017-08-09
"""

import pymysql
from alexutil import configutil
from alexutil import dbutil
from alexutil import dateutil
import traceback

conf_path = 'E://workspace/dbconf'

# db
online_host = configutil.get_value(conf_path, 'dbonline', 'host')
online_port = configutil.get_value(conf_path, 'dbonline', 'port')
online_user = configutil.get_value(conf_path, 'dbonline', 'user')
online_pass = configutil.get_value(conf_path, 'dbonline', 'pass')

online_conf_dict = dict(host = online_host, port = int(online_port), user = online_user, password = online_pass, db='recsys', charset = 'utf8')

def get_docid_from_videoarticle(start_time, end_time):
    docid_list = []
    with dbutil.connect(**online_conf_dict) as conn:
        try:
            with conn.cursor() as cursor:
                sql_str = 'select docid from video_article where insert_time >= "{}" and insert_time < "{}"'.format(start_time, end_time)
                print(sql_str)
                cursor.execute(sql_str)
                results = cursor.fetchall()

                for docid in results:
                    docid_list.append(docid[0])

        except Exception as e:
            print(e)
            traceback.print_exc()
    return docid_list

if __name__ == '__main__':
    # test_db()
    docid_list = get_docid_from_videoarticle('2017-08-08 00:00:00','2017-08-09 00:00:00')
    print(len(docid_list))
    print(docid_list[3])