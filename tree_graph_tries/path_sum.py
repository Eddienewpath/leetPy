def hasPathSum(root, sum):
    if not root: return False
    if not root.left and not root.right and sum - root.val == 0: return True 
    return hasPathSum(root.left, sum-root.val) or hasPathSum(root.right, sum-root.val)



# edge case: null tree and sum = 0 
# key: check at leaf level





#       5
#      / \
#     4   8
#    /   / \
#   11  13  4
#  /  \      \
# 7    2      1
# return true, as there exist a root-to-leaf path 5->4->11->2
    # [1,2] 1 