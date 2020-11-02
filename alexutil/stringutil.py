# coding: utf-8
'''
Created on 2017-05-16
@author:Alex Wang
字符串处理
'''


def to_str(bytes_or_str):
    if isinstance(bytes_or_str, bytes):
        return bytes_or_str.decode('utf-8')
    else:
        return bytes_or_str


def to_bytes(bytes_or_str):
    if isinstance(bytes_or_str, str):
        return bytes_or_str.encode('utf-8')
    else:
        return bytes_or_str


def check_if_is_chinese_or_digit(char):
    if (char > u'\u4e00' and char < u'\u9fff') or u'0' <= char <= '9':
        print(char)
        return True
    else:
        return False


def test():
    print("test")
    print(to_bytes("kdjfkd"))
    print(to_str(b'dsfjksdl'))


if __name__ == '__main__':
    string = u"./data/NCDC/上海/虹桥/6240476818161dat.txt"
    for char in string:
        print(check_if_is_chinese_or_digit(char))
