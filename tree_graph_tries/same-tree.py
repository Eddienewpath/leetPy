# Input:     1         1
#           /           \
#          2             2

#         [1,2],     [1,null,2]

# Output: false


# structurally identical and have same value
# meaning same traversal and same value
def isSameTree(p, q):
    if not p and not q: return True
    if not p or not q: return False
    return isSameTree(p.left, q.left) and p.val == q.val and isSameTree(p.right, q.right)


def isSameTree_iter(p, q):
    pass
