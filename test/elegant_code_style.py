"""
Created by Alex wang
on 20170512
"""


def ifelse(weight):
    body = "fat" if weight > 120 else "thin"
    print(body)


def test_cnumerate():
    print("test enumerate.........")
    str_list = ["one", "two", "three", "four"]
    for i, str in enumerate(str_list):
        print("{}\t{}".format(i, str))


def test_zip():
    print("test zip..........")
    list_one = [1, 2, 3, 4]
    list_two = ["one", "two", "three", "four"]
    for i, str in zip(list_one, list_two):
        print("{}\t{}".format(i, str))

    print("test zip..........")
    tuple_one = (1, 2, 3, 4)
    tuplt_two = ("one", "two", "three", "four")
    for i, str in zip(tuple_one, tuplt_two):
        print("{}\t{}".format(i, str))


def test_unzip():
    print("test unzip.........")
    tuple_list = [(1, 2), (1, 2), (1, 2)]
    a, b = zip(*tuple_list)
    print(a)
    print(b)


def test_join():
    li = ["one", "two", "three", "four", "five"]
    print(",".join(li))


def test_read():
    """
    读取文件
    :return:
    """
    count = 1
    file_path = "E://temp/code/alarm.py"
    for line in open(file_path, 'r', encoding='UTF-8'):
        count += 1
    print(count)

    with open(file_path, 'r', encoding='UTF-8') as reader:
        for line in reader:
            count += 1
    print(count)


def test_num():
    str = "2087"
    int_num = int(str)
    print(int_num)


def test_map():
    a = [1, 2, 3, 4, 5]
    print('test_map:', ','.join(map(str, a)))

if __name__ == "__main__":
    test_num()
    # test_read()
    test_unzip()
    test_map()
