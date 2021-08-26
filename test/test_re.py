"""
 @author: AlexWang
 @date: 2021/8/26 8:16 PM

 测试正则表达式

re.search()：若string中包含pattern子串，则返回Match对象，否则返回None，注意，如果string中存在多个pattern子串，只返回第一个。
re.match(): 从首字母开始开始匹配，string如果包含pattern子串，则匹配成功，返回Match对象，失败则返回None，若要完全匹配，pattern要以$结尾。

re.findall()：返回string中所有与pattern相匹配的全部字串，返回形式为数组。
re.finditer()：返回string中所有与pattern相匹配的全部字串，返回形式为迭代器。
"""

import re


def test_findAll():
    string = "aAlex9991111Alex991212"
    pattern = r'(Al[a-z]*)[0-9]*'
    result = re.findall(pattern, string)
    print(result)  # ['Alex', 'Alex']


def test_match():
    string = "Alex9991111Alex991212"
    pattern = r'(Alex[0-9]*)'

    result = re.match(pattern, string)
    print(result)  # <_sre.SRE_Match object; span=(0, 11), match='Alex9991111'>
    print(result.group())  # Alex9991111

    # --------------------------
    string = "aAlex9991111Alex991212"
    result = re.match(pattern, string)
    print(result)  # None


if __name__ == '__main__':
    # test_match()
    test_findAll()
