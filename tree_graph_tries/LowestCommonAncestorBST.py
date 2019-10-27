
def lowestCommonAncestor(root, p, q):
    if root.val < p.val and root.val < q.val: 
        return lowestCommonAncestor(root.right, p, q)
    if root.val > p.val and root.val > q.val: 
        return lowestCommonAncestor(root.left, p, q)
    else:
        return root


def lowestCommonAncestor_iter(root, p, q):
    pass
    
