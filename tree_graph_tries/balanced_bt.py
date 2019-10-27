class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


def isBalanced(root):
    if not root: return True
    def depth(root): 
        if not root: return -1
        return max(depth(root.left), depth(root.right))+1 
    return isBalanced(root.left) and isBalanced(root.right) and abs(depth(root.left)-depth(root.right)) <= 1


def isBalanced_iter(root):
    pass 


# Given the following tree[3, 9, 20, null, null, 15, 7]:

#     3
#    / \
#   9  20
#     /  \
#    15   7
# true 
