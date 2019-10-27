import copy
# s: "cbaebabacd" p: "abc"

# Output:
# [0, 6]

# brute force.  TLE
# def findAnagrams(s, p):
#     def check_anagram(st, chars):
#         for c in st:
#             if st.count(c) != chars.count(c):
#                 return False
#         return True
#     # check ahead for the length of p
#     res = []
#     p_len = len(p)
#     for i in range(len(s) - p_len+1):
#         if check_anagram(s[i : i+p_len], p): 
#             res.append(i)
#     return res 



# sliding window
def findAnagrams(s, p):
    n = len(p)
    dic = [0]*26
    res = []
    for c in p:
        dic[ord(c) - ord('a')] += 1 
        
    left, right, count = 0, 0, n 
    while right < len(s):
        if dic[ord(s[right])-ord('a')] > 0:
            count -= 1
        dic[ord(s[right])-ord('a')] -= 1         
        right += 1

        # if the count == 0 meaning all the dic freq also got reduce to 0 because the total of the freq is count
        if count == 0: res.append(left)

        if right - left == n: 
            if dic[ord(s[left])-ord('a')] >= 0: 
                count += 1
            dic[ord(s[left])-ord('a')] += 1
            left += 1 

    return res
            


print(findAnagrams('cbaebabacd', 'abc'))
