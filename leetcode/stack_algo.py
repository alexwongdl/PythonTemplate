"""
 @author: AlexWang
 @date: 2022/5/31 10:49 AM
 @Email: wangjian90@hotmail.com
"""

"""
用两个栈实现队列
用两个栈实现一个队列。队列的声明如下，请实现它的两个函数 appendTail 和 deleteHead ，分别完成在队列尾部插入整数和在队列头部删除整数的功能。
(若队列中没有元素，deleteHead 操作返回 -1 )

提示：
1 <= values <= 10000
最多会对 appendTail、deleteHead 进行 10000 次调用

思路：
一个队列用来存储数据，另一个队列用来辅助deleteHead
"""


class CQueue:

    def __init__(self):
        self.stack1 = []
        self.stack2 = []

    def appendTail(self, value: int) -> None:
        self.stack1.append(value)
        return None

    def deleteHead(self) -> int:
        if len(self.stack1) <= 0:
            return -1
        while len(self.stack1) > 1:
            item = self.stack1.pop()
            self.stack2.append(item)
        head = self.stack1.pop()

        while len(self.stack2) > 0:
            item = self.stack2.pop()
            self.stack1.append(item)
        return head


def test_cqueue():
    queue = CQueue()
    queue.appendTail(3)
    queue.deleteHead()
    queue.deleteHead()

    queue.appendTail(5)
    queue.appendTail(2)
    queue.deleteHead()
    queue.deleteHead()


"""
*****  包含min函数的栈
定义栈的数据结构，请在该类型中实现一个能够得到栈的最小元素的 min 函数在该栈中，调用 min、push 及 pop 的时间复杂度都是 O(1)。

解题思路：
我们只需要设计一个数据结构，使得每个元素 a 与其相应的最小值 m 时刻
保持一一对应。因此我们可以使用一个辅助栈，与元素栈同步插入与删除，用于存储与每个元素对应的
最小值。
"""


class MinStack:

    def __init__(self):
        self.stack = []
        self.min_stack = []
        self.top_val = None

    def push(self, val: int) -> None:
        if len(self.stack) <= 0:
            self.stack.append(val)
            self.min_stack.append(val)
            self.top_val = val
        else:
            self.stack.append(val)
            cur_min = self.min_stack[-1]
            if val < cur_min:
                self.min_stack.append(val)
            else:
                self.min_stack.append(cur_min)

        return None

    def pop(self) -> None:
        self.stack.pop()
        self.min_stack.pop()
        return None

    def top(self) -> int:
        return self.stack[-1]

    def min(self) -> int:
        return self.min_stack[-1]


def test_min_stack():
    min_stack = MinStack()
    min_stack.push(-2)
    min_stack.push(0)
    min_stack.push(-3)
    print(min_stack.min())  # --> 返回 -3.
    min_stack.pop()
    print(min_stack.top())  # --> 返回 0.
    print(min_stack.min())  # --> 返回 -2.


if __name__ == '__main__':
    # test_cqueue()
    test_min_stack()
