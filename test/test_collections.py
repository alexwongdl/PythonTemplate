"""
Created by Alex Wang on 20170704
测试Python内置类collections：containers--dict、list、set、tuple
"""
import collections

##TODO：测试collections内各个数据结构的特性
def test_namedtuple():
    print("test namedtuple...")

def test_deque():
    print("test deque...")

def test_chainmap():
    print("test chainmap...")

def test_counter():
    print("test counter...")
    counter_one = collections.Counter("sdfajldjfd")
    print(counter_one)


def test_ordereddict():
    print("test ordereddict...")

def test_defaultdict():
    print("test defaultdict...")

def test_userdict():
    print("test userdict...")

def test_userlist():
    print("test userlist...")

def test_userstring():
    print("test userstring...")

if __name__ == "__main__":
    test_counter()