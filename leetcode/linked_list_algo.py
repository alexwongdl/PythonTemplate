"""
 @author: AlexWang
 @date: 2022/5/25 3:58 PM
 @Email: alex.wj@alibaba-inc.com
"""


def print_nodes(head):
    list_a = [head.val]
    cur_node = head.next
    while cur_node:
        list_a.append(cur_node.val)
        cur_node = cur_node.next
    print(list_a)


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


"""
合并两个有序链表  Easy

将两个升序链表合并为一个新的 升序 链表并返回。新链表是通过拼接给定的两个链表的所有节点组成的。 

输入：l1 = [1,2,4], l2 = [1,3,4]
输出：[1,1,2,3,4,4]

输入：l1 = [], l2 = []
输出：[]

输入：l1 = [], l2 = [0]
输出：[0]
"""


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


def mergeTwoLists(list1, list2):
    if len(list1) <= 0:
        return list2
    elif len(list2) <= 0:
        return list1

    new_list = []
    list1_point = 0
    list2_point = 0

    while list1_point < len(list1) and list2_point < len(list2):
        list1_item = list1[list1_point]
        list2_item = list2[list2_point]
        if list1_item <= list2_item:
            new_list.append(list1_item)
            list1_point += 1
        else:
            new_list.append(list2_item)
            list2_point += 1

    while list1_point < len(list1):
        new_list.append(list1[list1_point])
        list1_point += 1

    while list2_point < len(list2):
        new_list.append(list2[list2_point])
        list2_point += 1
    print(new_list)
    return new_list


def test_merge_two_lists():
    l1 = [1, 2, 4]
    l2 = [1, 3, 4]
    mergeTwoLists(l1, l2)


"""
***** 排序链表
给你链表的头结点 head ，请将其按 升序 排列并返回 排序后的链表 。

输入：head = [4,2,1,3]
输出：[1,2,3,4]

输入：head = [-1,5,3,4,0]
输出：[-1,0,3,4,5]

输入：head = []
输出：[]

进阶：你可以在 O(n log n) 时间复杂度和常数级空间复杂度下，对链表进行排序吗？
"""


class SortListSolution:
    def sortList(self, head):
        if head is None or head.next is None:
            return head
        block_list = []
        while head:
            cur_node = head
            head = head.next
            cur_node.next = None
            block_list.append(cur_node)

        # 两两进行归并
        pivot = int(len(block_list) / 2)
        block_a = block_list[:pivot]
        block_b = block_list[pivot:]
        result = self.iter_merge(block_a, block_b)

        # print_nodes(result[0])
        return result[0]

    def iter_merge(self, block_a, block_b):
        # print(len(block_a), len(block_b))
        if len(block_a) >= 2:
            pivot = int(len(block_a) / 2)
            block_a_a = block_a[:pivot]
            block_a_b = block_a[pivot:]
            block_a = self.iter_merge(block_a_a, block_a_b)
        elif len(block_a) == 1:
            block_a = block_a

        if len(block_b) >= 2:
            pivot = int(len(block_b) / 2)
            block_b_a = block_b[:pivot]
            block_b_b = block_b[pivot:]
            block_b = self.iter_merge(block_b_a, block_b_b)
        elif len(block_b) == 1:
            block_b = block_b

        head_a = block_a[0]
        head_b = block_b[0]

        new_head = ListNode(0, None)
        cur_head = new_head
        while head_a and head_b:
            if head_a.val <= head_b.val:
                cur_head.next = head_a
                cur_head = head_a
                head_a = head_a.next
            else:
                cur_head.next = head_b
                cur_head = head_b
                head_b = head_b.next

        while head_a:
            cur_head.next = head_a
            cur_head = head_a
            head_a = head_a.next

        while head_b:
            cur_head.next = head_b
            cur_head = head_b
            head_b = head_b.next

        cur_head = new_head.next
        new_head = ListNode(0, None)

        return [cur_head]


def test_sort_list():
    solution = SortListSolution()
    list_a = [4, 2, 1, 3]
    head = ListNode(4, None)
    cur_node = head
    for i in range(1, len(list_a)):
        new_node = ListNode(list_a[i], None)
        cur_node.next = new_node
        cur_node = cur_node.next
    print_nodes(head)

    solution.sortList(head)


"""
******* 环形链表 II
给定一个链表的头节点  head ，返回链表开始入环的第一个节点。 如果链表无环，则返回 null。

如果链表中有某个节点，可以通过连续跟踪 next 指针再次到达，则链表中存在环。 为了表示给定链表中的环，评测系统内部使用整数 pos 来表示链表尾连接到链表中的位置（索引从 0 开始）。如果 pos 是 -1，则在该链表中没有环。注意：pos 不作为参数进行传递，仅仅是为了标识链表的实际情况。

不允许修改 链表。

输入：head = [3,2,0,-4], pos = 1
输出：返回索引为 1 的链表节点
解释：链表中有一个环，其尾部连接到第二个节点。

进阶：你是否可以使用 O(1) 空间解决此题？
"""


class CircleLinkSolution:
    def detectCycle(self, head: ListNode) -> ListNode:
        pass


def test_circle_link():
    list_a = [3, 2, 0, -4]
    head = ListNode(3, None)
    cur_node = head

    for i in range(1, len(list_a)):
        new_node = ListNode(list_a[i], None)
        cur_node.next = new_node
        cur_node = cur_node.next
        if i == 1:
            circle_node = cur_node
        if i == 3:
            cur_node.next = circle_node

    # print_nodes(head)


if __name__ == '__main__':
    # test_merge_two_lists()
    # test_sort_list()
    test_circle_link()
