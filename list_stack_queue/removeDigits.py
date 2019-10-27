# greedy algorithm 
# building smallest monotonic increasing sequence. 
def removeKdigits(num, k):
    if k == len(num): return '0'
    stack = []
    for d in num: 
        # push smaller digits as left as possible by poping out bigger number k times
        while k and stack and stack[-1] > d: 
            stack.pop()
            k -= 1 
        stack.append(d)
   
    res = ''.join(stack)
    for i, c in enumerate(res):
        if c != '0':
            return res[i:-k or None] 
    return '0'
# 1234
print(removeKdigits('1111', 1))


a = '000123'
a.lstrip('0') # output '123' left strip will remove leading chars which you passed in as argument

# if a b have same digits: whichever has greatest leftmost digit is greater
# if a b have same left most, move right a digit and compare

