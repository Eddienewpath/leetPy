import collections
def levelOrderBottom(root):
    res = []
    if not root: return res
    queue = collections.deque()
    queue.add(root)

    while queue: 
        size = len(queue)
        tmp = []
        for i in range(size):
            front = queue.popleft()
            tmp.append(front.val)
            if front.left: 
                queue.append(front.left)
            if front.right:
                queue.append(front.right)
        res.insert(0, tmp)
    return res
