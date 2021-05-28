import time
def fib_naive(n):
    if n == 1 or n == 2: 
        return 1
    return fib(n-1) + fib(n-2)

# start = time.time()
# print(fib_naive(35))
# print(round((time.time() - start)*1000))


print('memo optimize----------Top down-starts from bigger number')

def fib_memo(n, dic):
    if n == 1 or n == 2: return 1
    if n in dic and dic[n] > 0: return dic[n]

    dic[n] = fib_memo(n-1, dic) + fib_memo(n-2, dic)
    return dic[n]


start = time.time()
dic = {}
print(fib_memo(200, dic))
print((time.time() - start)*1000)


print('DP--------botton up')


def fib_dp(n):
    dp = [1] * (n+1)
    for i in range(3, n+1):
        dp[i] = dp[i-1] + dp[i-2]
    return dp[n]


begin = time.time()
print(fib_dp(200))
print((time.time()-begin)*1000)

print('optimal------------ variable solution')
def fib(n):
    if n == 1 or n == 2: return 1
    prev = cur = 1
    for i in range(3, n+1): 
        tmp = prev + cur
        prev = cur
        cur = tmp
    return cur


begin = time.time()
print(fib(200))
print((time.time()-begin)*1000)
