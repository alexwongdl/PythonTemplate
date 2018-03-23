"""
Created by Alex Wang on 20170704
测试Python内置类collections：containers--dict、list、set、tuple
"""
from collections import Counter
from collections import OrderedDict
from collections import namedtuple

##TODO：测试collections内各个数据结构的特性
def test_namedtuple():
    print("test namedtuple...")
    Person = namedtuple(typename='person', field_names=['name', 'age'])
    alex = Person(name = 'Alex Wang', age = '18')
    alex = alex._replace(age = '20')
    print(alex)

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

    print('test counter for word count...')
    word_counter = Counter()
    for line in open('dict_list_tuple_set.py', 'r', encoding='utf-8'):
        elems = line.split(' ')
        for elem in elems:
            word_counter.update({elem.strip(), 1})
    print('size of word_counter:', len(word_counter.keys()))
    del word_counter['1']
    del word_counter['=']
    most_common_words = word_counter.most_common(3)
    for key, cnt in most_common_words:
        print(key, cnt)

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


def test_dict_sort():
    """
    字典按照value值排序
    :return:
    """
    dict_one = {'a': 5, 'b': 10, 'c': 3, 'd': 6, 'e': 6}
    print(dict_one.items())
    sorted_one = sorted(dict_one.items(), key=lambda x: x[1], reverse=True)
    print([x[0] for x in sorted_one])


if __name__ == "__main__":
    test_counter()
    test_ordereddict()
    test_dict_sort()
    test_namedtuple()
