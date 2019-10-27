def lowestCommonAncestor(root, p, q):
    if not root or root.val == p.val or root.val == q.val: 
        return root
    left = lowestCommonAncestor(root.left, p, q)
    right = lowestCommonAncestor(root.right, p, q)
    if left and right: return root
    if not left and not right: return None
    if left: return left
    if right: return right




# LCA
# if p, q is under the tree rooted r, return LCA
# if neither p and q is under tree rooted r, return null 
# if p or q under the tree rooted r, return p or q
# bottom -> up and pass up the LCA


def lowestCommonAncestor_iter(root, p, q):
    stack = [root]
    parent = {root: None}

    while p not in parent or q not in parent: 
        top = stack.pop()

        if top.left: 
            parent[top.left] = top
            stack.append(top.left)

        if top.right: 
            parent[top.right] = top
            stack.append(top.right)

    # Create a set to hold all the ancestors of p for constant time access
    ancestors = set()
    
    # add all the precedents of p 
    while p:
        ancestors.add(p)
        p = parent[p]

    # find the first accestor that share with p 
    while q not in ancestors: 
        q = parent[q]
    
    return q 
