import collections
def rightSideView(root):
    ans = []
    if not root: return ans
    queue = collections.deque()
    queue.append(root)
    while queue: 
        size = len(queue)
        tmp = []
        while size: 
            front = queue.popleft()
            tmp.append(front.val)
            if front.left:
                queue.append(front.left)
            if front.right:
                queue.append(front.right)
            size -= 1
        ans.append(tmp[-1])
    return ans
            





# Input: [1, 2, 3, null, 5, null, 4]
# Output: [1, 3, 4]
# Explanation:

#    1            < ---
#  /   \
# 2     3         <---
#  \     \
#   5     4       <---
# level order last item
