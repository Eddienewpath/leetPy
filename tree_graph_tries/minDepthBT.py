def minDepth(root):
    if not root:
        return 0
        left = minDepth(root.left)
        right = minDepth(root.right)
        if left == 0 or right == 0:
            return left+right+1
        else:
            return min(left, right)+1





# key: leaf level for example edge case [1,2] result is 2 not 0
# root node itself is not leaf if it has children
#  
# Given binary tree[3, 9, 20, null, null, 15, 7],

#     3
#    / \
#   9  20
#     /  \
#    15   7
# my solution:
# if not root:return 0
 # if not root.left or not root.right:
        #     return max(self.minDepth(root.left), self.minDepth(root.right)) + 1
        # return min(self.minDepth(root.left), self.minDepth(root.right)) + 1
