def coin_change(coins, amount):
    if amount < 0: 
        return float('inf')
    if amount == 0:
        return 0

    min_cnt = float('inf')
    for c in coins:
        min_cnt = min(min_cnt, coin_change(coins, amount-c)+1)
    return min_cnt


print(coin_change([1,2,5], 11))



print('top-dowm-memo--------------')


def coin_change(coins, amount):
    def coins_helper(coins, amount, dic):
        if amount in dic and dic[amount] > 0: return dic[amount]
        if amount < 0:
            return float('inf')
        if amount == 0:
            return 0

        min_cnt = amount
        for c in coins:
            min_cnt = min(min_cnt, coins_helper(coins, amount-c, dic)+1)
        dic[amount] = min_cnt
        return min_cnt
    return coins_helper(coins, amount, {})


print(coin_change([1, 2, 5], 11))



print('bottom-up-dp-------------')

# dp[n]: mim number of coins to get n amount
def coin_change_dp(coins, amount):
    dp = [float('inf')] * (amount+1)
    dp[0] = 0
    for i in range(1, amount+1):
        for c in coins:
            if i - c >= 0:
                dp[i] = min(dp[i], dp[i-c] + 1)
    return dp[amount] if dp[amount] != float('inf') else -1


print(coin_change_dp([1, 2, 5], 11))
