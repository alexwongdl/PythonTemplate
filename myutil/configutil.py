'''
Created on 2017-5-17
@author: Alex Wang

Config文件解析模块，config文件格式如下，中括号[]内是section，section内是key-value的配置neirong
example:
[db]
db_host = 127.0.0.1
db_port = 22
db_user = root
db_pass = rootroot

[concurrent]
thread = 10
processor = 20
'''

import configparser

def get_value(path, section, key):
    config = configparser.ConfigParser()
    config.read(path)
    return config.get(section, key)