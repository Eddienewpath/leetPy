from collections import deque


class Solution:
    def zigzagLevelOrder(self, root: TreeNode) -> List[List[int]]:
        if not root:
            return []
        res, queue, flag = [], deque([root]), True

        while queue:
            tmp = []
            size = len(queue)
            while size:
                front = queue.popleft()
                if front.left:
                    queue.append(front.left)
                if front.right:
                    queue.append(front.right)
                if flag:
                    tmp.append(front.val)
                else:
                    tmp.insert(0, front.val)
                size -= 1
            res.append(tmp)
            flag = not flag

        return res
