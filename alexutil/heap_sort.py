"""
Created by Alex Wang
On 2018-11-27
"""
import sys
import numpy as np


class MinHeapSort():
    def __init__(self, size):
        """
        :param size: size of min heap
        """
        self.size = size
        self.size_bk = size
        self.topItems = [None for i in range(size + 1)]

        min_value = sys.float_info.min  # 2.2250738585072014e-308
        self.topScores = np.array([min_value for i in range(size + 1)])  # 索引从1到size
        self.sorted_list = []

    def shift_root_down(self, item, score):
        """
        替换掉最小堆的root节点后，需要把root往下移
        :param item:
        :param score:
        :return:
        """
        i = 1
        half = self.size / 2
        while i <= half:
            child = 2 * i
            right_child = child + 1
            if right_child <= self.size and self.topScores[child] > self.topScores[right_child]:
                child = right_child

            if score < self.topScores[child]:
                break

            self.topItems[i] = self.topItems[child]
            self.topScores[i] = self.topScores[child]
            i = child

        self.topItems[i] = item
        self.topScores[i] = score

    def try_add_item(self, item, score):
        """
        把下一个项和最小堆的root做对比，如果大于root则替换
        :param item:
        :param score:
        :return:
        """
        if score > self.topScores[1]:
            self.topItems[1] = item
            self.topScores[1] = score
            self.shift_root_down(item, score)

    def sort(self):
        """
        依次从最小堆堆顶删除root，保存到array末尾
        :return:
        """
        i = 0
        num = self.size
        while i < num:
            i += 1
            temp_item = self.topItems[1]
            temp_score = self.topScores[1]

            self.topItems[1] = self.topItems[self.size]
            self.topScores[1] = self.topScores[self.size]

            self.topItems[self.size] = temp_item
            self.topScores[self.size] = temp_score

            self.size -= 1
            self.shift_root_down(self.topItems[1], self.topScores[1])

        for i in range(1, num + 1):
            self.sorted_list.append((self.topItems[i], self.topScores[i]))
        return self.sorted_list

    def get_sorted_list(self):
        return self.sorted_list

    def print_status(self):
        str_item = ""
        str_score = ""
        for i in range(1, self.size_bk + 1):
            str_item += "{}".format(self.topItems[i]).center(10)
            str_score += "{:.2f}".format(self.topScores[i]).center(10)
        print(str_item)
        print(str_score)

if __name__ == '__main__':
    print("{}".format('abc').center(10) + "{}".format('efgh').center(10))
    print("{:.2f}".format(0.878456).center(10))

    item_list = ['a', 'bc', 'efg', 'h', 'j', 'kef', 'ghi']
    score_list = [0.9, 0.5, 0.4, 0.5, 0.3, 0.8, 0.2]

    minHeapSort = MinHeapSort(5)
    for i in range(len(item_list)):
        minHeapSort.try_add_item(item_list[i], score_list[i])
        minHeapSort.print_status()

    print('sort...')
    minHeapSort.sort()
    minHeapSort.print_status()
