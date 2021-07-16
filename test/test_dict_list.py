"""
 @author: AlexWang
 @date: 2021/7/15 8:54 PM
 @Email: alex.wj@alibaba-inc.com
"""

import ast


def string_to_list():
    """
    string 快速转成 list
    :return:
    """
    string = '[1, 2, 3]'
    arr = ast.literal_eval(string)
    print(arr)


if __name__ == '__main__':
    string_to_list()
