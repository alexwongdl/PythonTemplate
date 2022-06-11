"""
 @author: AlexWang
 @date: 2022/6/7 12:04 PM
 @Email: alex.wj@alibaba-inc.com
"""


class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


"""
从上到下打印二叉树 II
从上到下按层打印二叉树，同一层的节点按从左到右的顺序打印，每一层打印到一行。

例如:
给定二叉树: [3,9,20,null,null,15,7],

    3
   / \
  9  20
    /  \
   15   7
返回其层次遍历结果：

[
  [3],
  [9,20],
  [15,7]
]
"""
import queue


def levelOrder(root: TreeNode) -> list[list[int]]:
    q1 = queue.Queue()
    q2 = queue.Queue()
    if root is None:
        return []

    q1.put(root)
    q_list = [q1, q2]
    queue_id = 0
    result = []
    cur_result = []

    while not (q1.empty() and q2.empty()):
        cur_node = q_list[queue_id].get()
        cur_result.append(cur_node.val)

        if cur_node.left:
            q_list[1 - queue_id].put(cur_node.left)
        if cur_node.right:
            q_list[1 - queue_id].put(cur_node.right)
        if q_list[queue_id].empty():
            result.append(cur_result)
            cur_result = []
            queue_id = 1 - queue_id

    print(result)
    return result


def test_level_order():
    head = TreeNode(1)
    head.left = TreeNode(2)
    head.right = TreeNode(3)
    head.left.left = TreeNode(4)
    head.left.right = TreeNode(5)
    print(levelOrder(head))


"""
****** 树的子结构
输入两棵二叉树A和B，判断B是不是A的子结构。(约定空树不是任意一个树的子结构)
B是A的子结构， 即 A中有出现和B相同的结构和节点值。
例如:
给定的树 A:
     3
    / \
   4   5
  / \
 1   2
给定的树 B：
   4 
  /
 1
返回 true，因为 B 与 A 的一个子树拥有相同的结构和节点值。

示例 1：
输入：A = [1,2,3], B = [3,1]
输出：false
示例 2：
输入：A = [3,4,5,1,2], B = [4,1]
输出：true

优化方法：动态规划
        类似于最长公共子串，计算两个串的最长匹配，然后和B树大小做对比
"""
import queue


def isSubStructure(A: TreeNode, B: TreeNode) -> bool:
    A_head = A
    B_head = B
    if A is None or B is None:
        return False

    q = queue.Queue()
    q.put(A_head)
    while not q.empty():
        cur_a_head = q.get()
        if cur_a_head.val == B_head.val:
            # 两棵树同时往下走
            q_a = queue.Queue()
            q_b = queue.Queue()
            q_a.put(cur_a_head)
            q_b.put(B_head)

            match = True
            while not q_b.empty():
                b_cur = q_b.get()
                if q_a.empty():
                    match = False
                    break
                a_cur = q_a.get()
                print(b_cur.val, a_cur.val)
                if b_cur.val != a_cur.val:
                    match = False
                    break
                if b_cur.left:
                    if not a_cur.left:
                        match = False
                        break
                    else:
                        q_b.put(b_cur.left)
                        q_a.put(a_cur.left)
                if b_cur.right:
                    if not a_cur.right:
                        match = False
                        break
                    else:
                        q_b.put(b_cur.right)
                        q_a.put(a_cur.right)

            if match:
                return True

        if cur_a_head.left:
            q.put(cur_a_head.left)
        if cur_a_head.right:
            q.put(cur_a_head.right)

    return False


if __name__ == '__main__':
    test_level_order()