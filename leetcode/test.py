class Node():
    def __init__(self, val):
        self.val = val
        self.next = None


def invert(head):
    if head is None:
        return head

    prev = head
    cur_point = head.next
    if cur_point is None:
        return head

    next_point = cur_point.next

    if next_point is None:
        head.next = None
        cur_point.next = head
        return cur_point

    head.next = None
    while cur_point:
        # print(prev.val, cur_point.val, next_point.val)
        cur_point.next = prev

        # result = cur_point
        # while result:
        #     print(result.val)
        #     result = result.next

        prev = cur_point
        cur_point = next_point
        if next_point:
            next_point = next_point.next
        else:
            break
    return prev


arr = [1, 2, 3, 4]
head = Node(arr[0])
cur_node = head
for i in range(1, len(arr)):
    cur_node.next = Node(arr[i])
    cur_node = cur_node.next

result = invert(head)
while result:
    print(result.val)
    result = result.next
