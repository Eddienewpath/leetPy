def romanToInt(s):
    mapping = {'I': 1, 'V': 5, 'X':10, 'L': 50, 'C':100,'D':500, 'M':1000}
    total = 0
    for i in range(1, len(s)+1):
        if i == len(s) or mapping[s[i]] <= mapping[s[i-1]]:
            total += mapping[s[i-1]]
        else:
            total -= mapping[s[i-1]]
    return total

# Input: "MCMXCIV"
# Output: 1994
# Explanation: M = 1000, CM = 900, XC = 90 and IV = 4.


print(romanToInt("MCMXCIV"))
