class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None



def sortedArrayToBST(nums):
    def helper(left, right, nums):
        if left > right:
            return None
        mid = (left + right)//2
        root = TreeNode(nums[mid])
        root.left = helper(left, mid-1, nums)
        root.right = helper(mid+1, right, nums)
        return root
    return helper(0, len(nums)-1, nums)


def sortedArrayToBST_iter(nums):
    pass 



# Given the sorted array: [-10, -3, 0, 5, 9],

# One possible answer is: [0, -3, 9, -10, null, 5], which represents the following height balanced BST:

#       0
#      / \
#    -3   9
#    /   /
#  -10  5

