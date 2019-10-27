def numTrees(n):
    table = [0]*(n+1)
    table[0] = table[1] = 1
    for i in range(2, n+1):
        for j in range(1, n+1):
            table[i] += table[j-1]*table[i-j]
    return table[n]

# Input: 3
# Output: 5
# Explanation:
# Given n = 3, there are a total of 5 unique BST's:

# given sorted sequence
# G(n) is number of unique bsts for length of n ordered sequence
# F(i, n) is number of unique bsts using [i] as root where 1<=i<=n 
# G(n) = F(1, n) + F(2, n) + ... F(i, n) + .. F(n-1, n) + F(n, n)
# F(i, n) = G(i-1)*G(n-i) for exemple:
# [1,2,3,4,5,6,7]  F(3, 7) find number of unique bsts rooted at 3 equals to the cartesion product of 
# number of unique subtrees left side of 3 and number of unique subtrees right side of 3 
# left side of 3 ,[1,2], can be denoted as G(2) and right side of 3 ,[4,5,6,7], can be denoted as G(4)
# thus F(i, n) = G(i-1)*G(n-i)
# so we can define subproblem as: G(n) = G(0)*G(n-1) + G(1)*G(n-2) ...G(n-2)*G(1) + G(n-1)*G(0)

print(numTrees(3))