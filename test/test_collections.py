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
    count_pairs =  sorted(counter_one.items(), key=lambda x:-x[1])
    words, _ = list(zip(*count_pairs))
    word_to_id = dict(zip(words, range(len(words))))
    id_to_word = dict(zip(range(len(words)), words))
    print(count_pairs)
    print(words)
    print(word_to_id)
    print(id_to_word)


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