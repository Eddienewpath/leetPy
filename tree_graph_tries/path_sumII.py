# top -> down
def pathSum(root, sum):
    def helper(root, sum, alist, tmp):
        if not root: return
        if not root.left and not root.right and sum-root.val == 0:
            tmp.append(root.val)
            alist.append(tmp[:])
            tmp.pop() # still has to pop when correct 
            return 
        tmp.append(root.val)
        helper(root.left, sum-root.val, alist, tmp)
        helper(root.right, sum-root.val, alist, tmp)
        tmp.pop()

    res = []
    helper(root, sum, res, [])
    return res
    

# button up
# add left and right list into result list and them insert current root value 
# to the front of the list
def path_sum(root, sum):
    # if null tree, return empty list
    if not root: return []
    if not root.left and not root.right and sum-root.val == 0:
        # return end node of the path that sum to taget sum
        return [[root.val]]

    tmp = path_sum(root.left, sum-root.val) + path_sum(root.right, sum-root.val)
    return [[root.val] + l for l in tmp]

