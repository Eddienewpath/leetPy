def inorderTraversal_iter(root):
    if not root: return []
    stack = []
    ans = []
    while stack or root: 
        while root:
            stack.append(root)
            root = root.left
        
        if stack: 
            root = stack.pop()
            ans.append(root.val)
            root = root.right # take care of right tree
    return ans




