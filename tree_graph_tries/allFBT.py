def allPossibleFBT(N):
    res = []
    if N % 2 == 0: return res 
    if N == 1: 
        res.append(TreeNode(0))
        return res
    # for all the possible roots
    for i in range(1, N, 2):
        # traverse down 
        left = allPossibleFBT(i)# left side list of FBT 
        right = allPossibleFBT(N-i-1) 
        for l in left:
            for r in right:
                root = TreeNode(0)
                root.left = l
                root.right = r 
                res.append(root)
    return res 


