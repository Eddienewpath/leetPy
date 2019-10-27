
# 8
#       10
#      /  \
#     5   - 3
#    / \    \
#   3   2   11
#  / \   \
# 3  -2   1

# find number of path sum to target, path must be downward 
def pathSum(root, tar):
    def pathSum_helper(r, dic, cur_sum, tar):
        if not r: return 0 
        cur_sum += r.val
        res = dic.get(cur_sum - tar, 0)
        dic[cur_sum] = dic.get(cur_sum, 0) + 1

        res += pathSum_helper(r.left, dic, cur_sum, tar) + pathSum_helper(r.right, dic, cur_sum, tar)
        dic[cur_sum] = dic.get(cur_sum) - 1 
        return res
    return pathSum_helper(root, {0:1}, 0, tar)
    