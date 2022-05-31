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
***** 最小栈
设计一个支持 push ，pop ，top 操作，并能在常数时间内检索到最小元素的栈。

实现 MinStack 类:

MinStack() 初始化堆栈对象。
void push(int val) 将元素val推入堆栈。
void pop() 删除堆栈顶部的元素。
int top() 获取堆栈顶部的元素。
int getMin() 获取堆栈中的最小元素。
"""
class MinStack:

    def __init__(self):


    def push(self, val: int) -> None:


    def pop(self) -> None:


    def top(self) -> int:


    def getMin(self) -> int:

def test_min_stack():
    


if __name__ == '__main__':
    # test_cqueue()
