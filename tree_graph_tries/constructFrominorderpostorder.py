def constructFromPrePost(pre, post):
    if not pre: return None
    root = TreeNode(pre[0])
    if len(pre) == 1: return root

    L = post.index(pre[1]) + 1
    root.left = constructFromPrePost(pre[1 : L+1], post[ : L])
    root.right = constructFromPrePost(pre[L+1 : ], post[L : -1])
    return root 





""" Intuition """
""" 
given preorder and postorder arrays to construct the tree, the first thing to 
think about is when to stop constructing one subtree and start the other one.
thinking about when to stop means that I need to know how many nodes in the
left subtree, then we can find out when to start contructing right subtree
the number of nodes in left subtree is a sequence of continous elements in 
given arrays 

according to the rules of preorder and postorder traversal, the left subtree root 
is at pre[1] in preoder array, and for post order, the root is process at
the last and the left subtree is process first. 
so left subtree in postorder array starts at postion 0 and ends at index of pre[1]. 
I can get the Length of left subtree by L = index(pre[1]) + 1
use this L, we can partition the array recursively to construct the tree. 
"""
