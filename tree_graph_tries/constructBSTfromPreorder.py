def bstFromPreorder(preorder):
    def helper(preorder):
        if not preorder:
            return None
        if len(preorder) == 1:
            return TreeNode(preorder[0])

        root = TreeNode(preorder[0])

        idx = -1
        flag = False
        for i in range(1, len(preorder)):
            if preorder[i] > preorder[0]:
                idx = i
                flag = True
                break
        if not flag:
            idx = len(preorder)

        root.left = helper(preorder[1: idx])
        root.right = helper(preorder[idx:])

        return root
    return helper(preorder)
