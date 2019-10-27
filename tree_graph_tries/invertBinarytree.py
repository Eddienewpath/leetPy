# Input:

#      4
#    /   \
#   2     7
#  / \   / \
# 1   3 6   9
# Output:

#      4
#    /   \
#   7     2
#  / \   / \
# 9   6 3   1

# for every tree swap its left and right child trees return root
# post order 
def invertTree_post(root):
    if root:
        left = invertTree(root.left)
        right = invertTree(root.right)
        root.left, root.right = right, left # you have connect root with the swapped node, can just simply swap nodes
        return root
    


# preorder 
def invertTree_pre(root):
    def helper(root):
        if not root: return
        left , right = root.left, root.right
        root.left, root.right = right, left
        helper(root.left)
        helper(root.right)
    helper(root)
    return root


# iterative post order 
def invertTree(root):
    if not root: return root
    r = root
    stack = [r]
    while stack:
        t = stack.pop()
        t.left, t.right = t.right, t.left
        if t.left:
            stack.append(t.left)
        if t.right: 
            stack.append(t.right)
    return root

