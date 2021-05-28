# memo
def longest_increasing_subsequence(nums):
    def dfs(nums, start, cur, dic):
        if start == len(nums):
            return 0 
        
        if nums[start] in dic and dic[nums[start]] > 0:
            return dic[nums[start]]
        
        mx = float('-inf')
        for i in range(start, len(nums)):
            if nums[i] > cur: 
                mx = max(mx, dfs(nums, i+1, nums[i], dic) + 1)
        dic[start] = mx
        return mx
    return dfs(nums, 0, float('-inf'), {})

nums = [10,9,2,5,3,7,101,18]
# print(longest_increasing_subsequence(nums))


print('dp-----------')
# prefix i solution below, alternative solution is LIS ended with [i] not implemented here
def longest_increasing_subsequence_dp(nums):
    dp = [1]*(len(nums)+1)
    dp[0] = 0
    for i in range(2, len(nums)+1):
        tmp = 0
        for k in range(1, i):
            if nums[i-1] > nums[k-1]:
                tmp = max(tmp, dp[k])
        dp[i] = tmp + 1
    return dp[-1]


print(longest_increasing_subsequence_dp(nums))
