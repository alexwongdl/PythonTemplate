"""
Created by hzwangjian1
on 2017-07-25
"""

import json
import os
import traceback

import pymysql

from ddzj.util import db
from myutil import configutil
from myutil import dateutil
from myutil import dbutil
from myutil import logutil

config_path = os.path.join(os.getcwd(), "config")
log_path = os.path.join(os.getcwd(), "info.log")
logger = logutil.LogUtil(log_path)

# db
online_host = configutil.get_value(config_path, "dbonline", "host")
online_port = int(configutil.get_value(config_path, "dbonline", "port"))
online_user = configutil.get_value(config_path, "dbonline", "user")
online_pass = configutil.get_value(config_path, "dbonline", "pass")
con_params = dict(host=online_host, user=online_user, password=online_pass,  # 线上
              database='recsys', port=online_port, charset='utf8', use_unicode=True)

conn = None

def insert_video_quality(result_list):
    """
    视频静态质量结果插入到video_quality_nlp
    :param result_list:
    :return:
    """
    # try:
    with dbutil.get_connect_cursor(**con_params) as (conn, cursor):
        date_str = dateutil.current_date_format()
        # 7,9,3
        sql='''
        insert ignore into video_quality_nlp(docid, quality_score, quality_level, no_audio, category, duration, mpix_unit_time,
        video_level, bit_rate, tid_level, definition, big_img, video_height, video_width, fm, resolution,
        blackedge, insert_day, others, title_cheat, content_richness, content_serverity_value, qr_code)
        values (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        '''
        tuple_list = []
        for result in result_list:
            other_info =  {'player_black_edge':result['player_black_edge'], 'cnn_title_cheat':result['cnn_title_cheat'],'kw_title_cheat':result['kw_title_cheat']}
            tuple_list.append((result['doc_id'], result['total_score'], result['total_score_level'], result['no_audio'], result['category'],
                               result['duration'], result['mpix_unit_time'], result['video_level'], result['bit_rate'], result['tid_level'],
                               result['definition'], result['contain_big_image'], result['video_height'], result['video_width'],
                               result['norm_mean_resolution'], result['resolution'], result['content_black_box_type'],
                               date_str,json.dumps(other_info), result['title_cheat'], result['static_video'], result['sansu_value'], result['qr_code']))
        cursor.executemany(sql, tuple_list)
        conn.commit()

            # except Exception as e:
            #     logger.error(e)
            #     traceback.print_exc()
            #     logger.error(traceback.format_exc())



def get_docid_from_videoarticle(start_time, end_time):
    """
    从video_article获取start_time和end_time之间插入的文章docid列表
    :param start_time:
    :param end_time:
    :return:
    """
    docid_list = []
    with dbutil.connect(**con_params) as conn:
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

def insert_video_quality_result(video_quality_list):
    """
    把视频静态质量结果插入到数据库
    :param video_quality_list:
    :return:
    """
    logger.info("insert records:{}".format(len(video_quality_list)))
    insert_video_quality(video_quality_list)


def get_videos(start_time, end_time):
    with db.get_cursor(**con_params) as cur:
        sql = '''
select
    a.mp4_url,
    a.m3u8_url,
    a.docid,
    a.interests,
    a.source_title,
    a.title,
    a.subtitle,
    a.content,
    a.category,
    a.tags,
    a.playersize,
    a.hits,
    a.location_type,
    a.pic_url,
    a.big_image_url,
    a.doc_url,
    a.publish_time,
    a.insert_time,
    a.expire_time,
    a.urls
from
    recsys.video_article a
where
    a.status >= 0
        and a.insert_time >= '{}' and a.insert_time < '{}' limit 22
    '''.format(start_time, end_time)
        cur.execute(sql)
        results = cur.fetchall()
        return results

def get_conn():
    global conn
    if conn is None:
        conn = pymysql.connect(**con_params)
    return conn

def get_mp4_url_by_doc_id(docid):
    mp4url = None
    with db.cursor(get_conn()) as cur:
        sql = "select a.mp4_url from recsys.video_article a where a.docid = '%s'" % docid
        cur.execute(sql)
        results = cur.fetchone()
        if results and len(results) >= 1:
            mp4url = results[0]
    return mp4url

def get_doc_by_doc_id(docid):
    mp4_url, doc_id, interests, source_title, title, category = [None] * 6
    with db.cursor(get_conn()) as cur:
        sql = "select " \
              "a.mp4_url," \
              "a.docid," \
              "a.interests," \
              "a.source_title," \
              "a.title," \
              "a.category" \
              " from recsys.video_article a where a.docid = '%s' limit 1" % docid
        cur.execute(sql)
        results = cur.fetchone()
        if results and len(results) == 6:
            mp4_url, doc_id, interests, source_title, title, category = results
    return mp4_url, doc_id, interests, source_title, title, category

if __name__ == "__main__":
    docid_list = get_docid_from_videoarticle('2017-08-08 00:00:00','2017-08-09 00:00:00')
    print(len(docid_list))
    print(docid_list[3])