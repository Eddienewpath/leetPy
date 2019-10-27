class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None



def constructMaximumBinaryTree(nums):
    def build_maxTree(start, end, nums):
        if start > end:
            return None
        max_num = float('-inf')
        pos = 0
        for i, n in enumerate(nums[start: end+1]):
            if n > max_num:
                pos = i+start
                max_num = n
        root = TreeNode(max_num)
        root.left = build_maxTree(start, pos-1, nums)
        root.right = build_maxTree(pos+1, end, nums)
        return root
    return build_maxTree(0, len(nums)-1, nums)


def constructMaximumBinaryTree_iter(nums):
    pass 

# Input: [3, 2, 1, 6, 0, 5]
# Output: return the tree root node representing the following tree:

#       6
#     /   \
#    3     5
#     \    / 
#      2  0   
#        \
#         1

# rules: 
# The root is the maximum number in the array.
# The left subtree is the maximum tree constructed from left part subarray divided by the maximum number.
# The right subtree is the maximum tree constructed from right part subarray divided by the maximum number.
