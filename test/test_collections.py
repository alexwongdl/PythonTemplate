"""
Created by Alex Wang on 20170704
测试Python内置类collections：containers--dict、list、set、tuple
"""
from collections import Counter
from collections import OrderedDict


##TODO：测试collections内各个数据结构的特性
def test_namedtuple():
    print("test namedtuple...")


def test_deque():
    print("test deque...")


def test_chainmap():
    print("test chainmap...")


def test_counter():
    print("test counter...")
    counter_one = Counter("sdfajldjfd")
    print(counter_one.most_common(3))
    count_pairs = sorted(counter_one.items(), key=lambda x: -x[1])
    words, _ = list(zip(*count_pairs))
    word_to_id = dict(zip(words, range(len(words))))
    id_to_word = dict(zip(range(len(words)), words))
    print(count_pairs)
    print(words)
    print(word_to_id)
    print(id_to_word)


def test_ordereddict():
    """
    Orderdict：保持插入式的顺序
    :return:
    """
    print("test ordereddict...")
    items = [("baidu", 1), ("alibaba", 2), ("tescent", 3)]
    order_dict = OrderedDict(items)
    for k, v in order_dict.items():
        print(k + "\t" + str(v))


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
    # test_ordereddict()
