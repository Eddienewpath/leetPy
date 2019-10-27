import collections
def findBottomLeftValue(root):
    if not root: return None
    res = []
    q = collections.deque()
    q.append(root)
    while q:
        size = len(q)
        tmp = []
        for _ in range(size):
            front = q.popleft()
            tmp.append(front.val)
            if front.left:
                q.append(front.left)
            if front.right:
                q.append(front.right)
        res.append(tmp)
    return res[-1][0]
