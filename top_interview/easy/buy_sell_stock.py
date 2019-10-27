# peak and valley
# The key point is we need to consider every peak immediately following a valley to maximize the profit.
# order matters, first find valley to buy in then find peak to sell.
def maxProfit(prices):
    max_profit = 0
    j = 0
    while j < len(prices):
        while j < len(prices) and (j == 0 or prices[j] <= prices[j-1]):
            j += 1

        valley = prices[j-1]

        while j < len(prices) and prices[j] > prices[j-1]:
            j += 1

        peak = prices[j-1]

        max_profit += peak - valley

    return max_profit


#  keep on adding the profit obtained from every consecutive transaction
def maxProfit_short(prices):
    max_profit = 0 
    for i in range(1, len(prices)):
        if prices[i] > prices[i-1]: 
            max_profit += prices[i] - prices[i-1]

    return max_profit

a = [7, 1, 5, 3, 6, 4]
# Output: 7
print(maxProfit_short(a))



