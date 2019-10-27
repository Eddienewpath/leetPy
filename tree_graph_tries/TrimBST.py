# return trimed tree root
def trimBST(root, L, R):
    if not root: return None
    if root.val < L: 
        return trimBST(root.right, L, R)
    if root.val > R: 
        return trimBST(root.left, L, R)
    else: 
        root.right = trimBST(root.right, L, R)
        root.left = trimBST(root.left, L, R)
        return root

def trimBST_iter(r, L, R):
    pass



# Input:
#     3
#    / \
#   0   4
#    \
#     2
#    /
#   1

#   L = 1
#   R = 3

# Output: 
#       3
#      / 
#    2   
#   /
#  1
