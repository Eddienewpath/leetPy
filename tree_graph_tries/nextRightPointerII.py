import collections
# bfs
def connect(root):
    if not root: return root 
    q = collections.deque()
    q.append(root)
    while q: 
        dummy = Node(-1, None, None, None)
        size = len(q)
        for _ in range(size):
            front = q.popleft()
            dummy.next = front
            dummy = dummy.next
            if front.left: q.append(front.left)
            if front.right: q.append(front.right)
    return root


    
