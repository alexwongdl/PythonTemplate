def convert_to_octal(input):
    base = 8
    result = 0
    multi = 1

    while input > base:
        input = input // base
        remin = input % base
        multi *= 10
        result = remin * multi + result
    print(result)
    return result


def test_convert_to_octal():
    convert_to_octal(10)
    convert_to_octal(1000000)


import numpy as np


def longest_path(matrix):
    if len(matrix) <= 0 or len(matrix[0]) <= 0:
        return 0
    rows = len(matrix)
    cols = len(matrix[0])

    result = np.zeros(shape=(rows, cols), dtype=np.int32)
    direct = np.zeros(shape=(rows, cols), dtype=np.int32)  # 1:left , 2:upper, 3:start
    for i in range(rows):
        for j in range(cols):
            if i <= 0 and j <= 0:
                result[i, j] = matrix[i][j]
            elif i <= 0:  # 只看left
                left = result[i, j - 1]
                if matrix[i][j] > matrix[i][j] + left:
                    result[i, j] = matrix[i][j]
                    direct[i, j] = 3
                else:
                    result[i, j] = matrix[i][j] + left
                    direct[i, j] = 1
            elif j <= 0:  # 只看upper
                upper = result[i - 1, j]
                if matrix[i][j] > matrix[i][j] + upper:
                    result[i, j] = matrix[i][j]
                    direct[i, j] = 3
                else:
                    result[i, j] = matrix[i][j] + upper
                    direct[i, j] = 2
            else:
                left = result[i, j - 1] + matrix[i][j]
                upper = result[i - 1, j] + matrix[i][j]

                if matrix[i][j] > left and matrix[i][j] > upper:
                    result[i, j] = matrix[i][j]
                    direct[i, j] = 3
                elif left > matrix[i][j] and left > upper:
                    result[i, j] = left
                    direct[i, j] = 1
                else:
                    result[i, j] = upper
                    direct[i, j] = 2

    i = rows - 1
    j = cols - 1
    num_list = []
    while True:
        num = matrix[i][j]
        num_list.insert(0, num)
        direct_cur = direct[i, j]
        if direct_cur == 0 or direct_cur == 3:
            break
        if direct_cur == 1:
            j -= 1
        else:
            i -= 1

    print(num_list)
    print(result[-1, -1])
    # print(result)
    # print(direct)


def test_longest_path():
    # matrix = [[1, 4, 3],
    #           [2, 3, 1],
    #           [2, 3, 4]]
    # longest_path(matrix)
    matrix = [[1, 1, 1, 1, 2],
              [2, 3, 4, 1, 4],
              [3, 1, 4, 2, 4],
              [2, 1, 5, 7, 2],
              [4, 3, 3, 4, 5]]
    longest_path(matrix)


def longest_path_1(matrix):
    if len(matrix) <= 0 or len(matrix[0]) <= 0:
        return 0
    rows = len(matrix)
    cols = len(matrix[0])

    result = np.zeros(shape=(rows, cols), dtype=np.int32)
    direct = np.zeros(shape=(rows, cols), dtype=np.int32)  # 1:left , 2:upper, 3:start
    result_2 = np.zeros(shape=(rows, cols), dtype=np.int32)
    direct_2 = np.zeros(shape=(rows, cols), dtype=np.int32)  # 1:left , 2:bottom, 3:start
    for i in range(rows):
        for j in range(cols):
            if i <= 0 and j <= 0:
                result[i, j] = matrix[i][j]
            elif i <= 0:  # 只看left
                left = result[i, j - 1]
                if matrix[i][j] > matrix[i][j] + left:
                    result[i, j] = matrix[i][j]
                    direct[i, j] = 3
                else:
                    result[i, j] = matrix[i][j] + left
                    direct[i, j] = 1
            elif j <= 0:  # 只看upper
                upper = result[i - 1, j]
                if matrix[i][j] > matrix[i][j] + upper:
                    result[i, j] = matrix[i][j]
                    direct[i, j] = 3
                else:
                    result[i, j] = matrix[i][j] + upper
                    direct[i, j] = 2
            else:
                left = result[i, j - 1] + matrix[i][j]
                upper = result[i - 1, j] + matrix[i][j]

                if matrix[i][j] > left and matrix[i][j] > upper:
                    result[i, j] = matrix[i][j]
                    direct[i, j] = 3
                elif left > matrix[i][j] and left > upper:
                    result[i, j] = left
                    direct[i, j] = 1
                else:
                    result[i, j] = upper
                    direct[i, j] = 2

    result_2[-1, :] = result[-1, :]
    result_2[:, 0] = result[:, 0]
    for i in range(rows - 2, -1, -1):
        for j in range(1, cols):
            left = result_2[i, j - 1] + matrix[i][j]
            bottom = result_2[i + 1, j] + matrix[i][j]

            if matrix[i][j] > left and matrix[i][j] > bottom:
                result_2[i, j] = matrix[i][j]
                direct_2[i, j] = 3
            elif left > matrix[i][j] and left > bottom:
                result_2[i, j] = left
                direct_2[i, j] = 1
            else:
                result_2[i, j] = bottom
                direct_2[i, j] = 2

    i = 0
    j = cols - 1
    num_list = []
    while True:
        num = matrix[i][j]
        num_list.insert(0, num)
        direct_cur = direct_2[i, j]
        if direct_cur == 0 or direct_cur == 3:
            break
        if direct_cur == 1:
            j -= 1
        else:
            i += 1
    num_list = num_list[1:]
    while True:
        num = matrix[i][j]
        num_list.insert(0, num)
        direct_cur = direct[i, j]
        if direct_cur == 0 or direct_cur == 3:
            break
        if direct_cur == 1:
            j -= 1
        else:
            i -= 1

    print(num_list)
    print(sum(num_list))
    # print(result)
    # print(direct)
    # print(result_2)
    # print(direct_2)
    # print("=================" * 2)


def test_longest_path_1():
    # matrix = [[1, 1, 1, 1],
    #           [2, 3, 2, 2],
    #           [2, 3, 4, 1]]
    # longest_path_1(matrix)
    matrix = [[1, 1, 1, 1, 2],
              [2, 3, 4, 1, 4],
              [3, 1, 4, 2, 4],
              [2, 1, 5, 7, 2],
              [4, 3, 3, 4, 5]]
    longest_path_1(matrix)


def self_dup_str(string):
    str_len = len(string)
    dup_str = ""

    for i in range(str_len):
        for j in range(i + 2, str_len):
            sub_str_len = j - i
            if j + sub_str_len > str_len:
                break
            succeed = True
            for k in range(sub_str_len):
                if string[i + k] != string[j + k]:
                    succeed = False
                    break
            if succeed:
                sub_str = string[i: i + sub_str_len * 2]
                if len(sub_str) > len(dup_str):
                    dup_str = sub_str
    print(dup_str)


def test_self_dup_str():
    # self_dup_str("abababcdabcd")
    self_dup_str("faaacabcddcbabcddcbedfgaac")


def connect_subgraph(matrix):
    rows = len(matrix)
    node_map = {}
    node_set_dict = {}

    if rows <= 1:
        return 0
    for i in range(rows):
        for j in range(i + 1, rows):
            if matrix[i][j] == 1:
                if i in node_map:
                    node_map[j] = node_map[i]
                    node_set_dict[node_map[j]].add(j)
                else:
                    node_map[j] = i
                    node_map[i] = i
                    node_set_dict[i] = set()
                    node_set_dict[i].add(i)
                    node_set_dict[i].add(j)

    print(len(node_set_dict.keys()))
    # print(node_set_dict)


def test_connect_subgraph():
    # matrix = [[0, 1, 1, 0, 0],
    #           [1, 0, 1, 0, 0],
    #           [1, 1, 0, 0, 0],
    #           [0, 0, 0, 0, 1],
    #           [0, 0, 0, 1, 0]]
    # connect_subgraph(matrix)
    matrix = [[0, 1, 0, 0, 0, 1, 0],
              [1, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 1, 1, 0, 0],
              [0, 0, 1, 0, 1, 0, 0],
              [0, 0, 1, 1, 0, 0, 0],
              [1, 0, 0, 0, 0, 0, 1],
              [0, 0, 0, 0, 0, 1, 0]]
    connect_subgraph(matrix)


def dup_str(string):
    if len(string) <= 1:
        print(string)

    longest_str = ""
    cur_longest = 0
    long_match = None
    long_dir = None
    longest_a = ""
    longest_b = ""

    for split_idx in range(1, len(string) - 1):
        part_a = string[0:split_idx]
        part_b = string[split_idx:]
        len_a = len(part_a)
        len_b = len(part_b)
        # 计算两个之间最长匹配距离
        match = np.zeros(shape=(len_a, len_b), dtype=np.int32)
        direct = np.zeros(shape=(len_a, len_b), dtype=np.int32)  # 上-1， 左-2， 左上-3

        for i in range(len_a):
            for j in range(len_b):
                cur_a = part_a[i]
                cur_b = part_b[j]
                if i == 0 or j == 0:
                    if cur_a == cur_b:
                        match[i, j] = 1
                else:
                    if cur_a == cur_b:
                        upper = match[i - 1, j]
                        left = match[i, j - 1]
                        corner = match[i - 1, j - 1] + 1
                    else:
                        upper = match[i - 1, j]
                        left = match[i, j - 1]
                        corner = match[i - 1, j - 1]
                    if upper > left and upper > corner:
                        match[i, j] = upper
                        direct[i, j] = 1
                    elif left > upper and left > corner:
                        match[i, j] = left
                        direct[i, j] = 2
                    else:
                        match[i, j] = corner
                        direct[i, j] = 3

        if match[-1, -1] > cur_longest:
            cur_longest = match[-1, -1]
            long_match = match
            long_dir = direct
            longest_a = part_a
            longest_b = part_b

    i = len(longest_a) - 1
    j = len(longest_b) - 1
    num_list = []
    while True:
        value_a = longest_a[i]
        value_b = longest_b[j]
        if value_a == value_b:
            num_list.insert(0, value_a)
        direct_cur = long_dir[i, j]
        if direct_cur == 1: # 上-1， 左-2， 左上-3
            i -= 1
        elif direct_cur == 2:
            j -=1
        elif direct_cur == 3:
            i -= 1
            j -= 1
        else:
            break

    print("".join(num_list))
    print(cur_longest)
    # print(long_match)
    # print(long_dir)
    # print(longest_a)
    # print(longest_b)


def test_dup_str():
    # dup_str("abababcccdaaaabcd")
    dup_str("fabzacabtcddcbabecdfdcbedfgaac")


if __name__ == '__main__':
    # test_convert_to_octal()
    # test_longest_path()
    # test_self_dup_str()
    # test_connect_subgraph()
    # test_longest_path_1()
    test_dup_str()
