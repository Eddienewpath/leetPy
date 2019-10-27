import collections
def levelOrder(root):
    if not root: return []
    queue = collections.deque()
    queue.append(root)
    ans = []

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
        ans.append(tmp)
    return ans

# Given binary tree[3, 9, 20, null, null, 15, 7],
#     3
#    / \
#   9  20
#     /  \
#    15   7
# return its level order traversal as:
# [
#   [3],
#   [9,20],
#   [15,7]
# ]

# preorder and add val when in same level
def levelOrder_recur(root):
    if not root: return []
    def helper(root , ans, level): 
        if not root: return
        # add new list  
        if len(ans) == level: 
            ans.append([])
        ans[level].append(root.val)
        helper(root.left, ans, level+1)
        helper(root.right, ans, level+1)

    ans = []
    helper(root, ans, 0)
    return ans

