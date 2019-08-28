# check palindrome
# find all the possible substrings

# abc [a, ab, abc, b, bc, c]
# find repeat operations 
# found first in last out relation so stack
# count
# stack '' <- top

def countSubstrings(s):
    res = 0
    for i in range(len(s)):
        for j in range(i, len(s)):
            cur = s[i:j+1]
            print(cur)
            if cur == cur[::-1]:
                res += 1 
    return res

print(countSubstrings('aba')) #6
