class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None



def buildTree(preorder, inorder):
    def construct(start, end, preorder, inorder, i):
        if i > len(preorder)-1 or start > end: return None
        root = TreeNode(preorder[i])
        idx = inorder.index(preorder[i])
        root.left = construct(start, idx-1, preorder, inorder, i+1)
        # idx - start = length of left subtree, i+idx-start+1 is the index of root of right subtree in preorder
        root.right = construct(idx+1, end, preorder,inorder, i+idx-start+1) 
        return root
    n = len(preorder)-1
    return construct(0, n, preorder, inorder, 0)
            
            


def buildTree_cache(preorder, inorder):
    pass


# For example, given

# preorder = [3, 9,20,15,7]
# inorder = [9, 3, 15, 20, 7]
# Return the following binary tree:

#     3
#    / \
#   9  20
#     /  \
#    15   7

# preorder at 0 pos is the root of the entire tree
# inorder: find [0] of preorder in inorder at index i, left subtree is form [0, i-1] i elements 
# and right tree [i+1, n]  n-i-1 elements 
