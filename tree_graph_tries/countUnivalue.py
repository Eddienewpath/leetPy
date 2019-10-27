def countUnivalSubtrees(root):
    count = [0]
    def helper(root, count):
        # null tree defined to be unival tree 
        if not root: return True
        # dfs
        left = helper(root.left, count) # whether or not left subtree is unival
        right = helper(root.right, count)
        # if left and right side both are unival, we only need to check if the r.v == r.l.v r.v == r.r.v
        if left and right: 
            # check all the conditons that make it false 
            if root.left and root.left.val != root.val: return False
            if root.right and root.right.val != root.val: return False
            # if above didnt return false, meaning when left and right are true, rest cases are ture
            count[0] += 1
            return True
        return False
    helper(root, count) 
    
    return count[0]




""" A Uni-value subtree means all nodes of the subtree have the same value.
count number of univalue subtrees 
 """


# Input:  root = [5, 1, 5, 5, 5, null, 5]

#               5
#              / \
#             1   5
#            / \   \
#           5   5   5

# Output: 4 





""" 
when doing if else type logic, list out all the conditions that makes True and all the condition
make false, then try to reduce the if else by looking for equavalence logic 
"""