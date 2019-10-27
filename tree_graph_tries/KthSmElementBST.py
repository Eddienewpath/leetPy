def kthSmallest(root, k):
    def helper(root, arr):
        if not root: return
        helper(root.left, arr)
        arr.append(root.val)
        helper(root.right, arr)
    res = []
    helper(root, res)
    return res[k-1]


# Input: root = [5, 3, 6, 2, 4, null, null, 1], k = 3
#        5
#       / \
#      3   6
#     / \
#    2   4
#   /
#  1
# Output: 3


def kthSmallest_iter(root, k):
    stack = []
    while root or stack:
        while root:
            stack.append(root)
            root = root.left
        if stack:
            root = stack.pop()
            k -= 1
            if k == 0:
                return root.val
            root = root.right

    return root.val



