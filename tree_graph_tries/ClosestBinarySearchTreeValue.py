def closestValue(root, target):
    close = root.val
    while root: 
        if abs(target - root.val) <  abs(target - close):
            close = root.val
        root = root.left if root.val > target else root.right
    return close







# Input: root = [4, 2, 5, 1, 3], target = 3.714286

#     4
#    / \
#   2   5
#  / \
# 1   3

# Output: 4
# def helper(root, res):
#             if not root: return

#             helper(root.left, res)
#             res.append(root.val)
#             helper(root.right, res)

#         res = []
#         helper(root, res)
#         print(res)
#         min_idx = 0
#         min_diff = float('inf')
#         for i, n in enumerate(res):
#             if abs(target - n) < min_diff: 
#                 min_idx = i
#                 min_diff = abs(target - n)
        
#         return res[min_idx]
