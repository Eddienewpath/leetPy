def countNodes(root):
    def helper(root):
        if not root: return 0
        return 1+helper(root.left)+helper(root.right)
    return helper(root)


