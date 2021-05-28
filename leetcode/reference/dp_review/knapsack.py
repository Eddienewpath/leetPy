# recursive -memo
# n items put into a bag capacity is w. each of the n item has value and weight.
# find the max value possible
def knapsack01(n, w, wt, val):
    def dfs(start, n, w, wt, val, dic):
        if n < 0 or w < 0: 
            return -1 
        if n == 0 or w == 0:
            return 0 
        
        if (n, w) in dic and dic[(n, w)] > 0:
            return dic[(n, w)]

        mx = 0
        for i in range(start, len(wt)):
            tmp = dfs(i+1, n-1, w-wt[i], wt, val, dic)
            if tmp < 0: continue 
            mx = max(mx, (tmp + val[i]))
        dic[(n,w)] = mx
        return mx
    return dfs(0, n, w, wt, val, {})


# print(knapsack01(3,4,[2,1,3],[4,2,3]))




def knapsack01_dp(n, w, wt, val):
    dp = [[0 for _ in range(w+1)] for _ in range(n+1)]

    for i in range(1, n+1):
        for j in range(1, w+1):
            if j - wt[i-1] < 0:
                dp[i][j] = dp[i-1][j]
            else:
                dp[i][j] = max(dp[i-1][j], dp[i-1][j-wt[i-1]]+val[i-1])
    return dp[-1][-1]


print(knapsack01_dp(3, 4, [2, 1, 3], [4, 2, 3]))
