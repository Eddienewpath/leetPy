# cnt = [0]
# cnt[0] += 1
# print(cnt[0])



# res = [4,1,2,3]

# for i in range(0, 10, 2):
#     print(i)

# a = [[1,2,3,4]]
# b = [[6,7,8,9]]
# # print(a + b)
# c = [[1,2,3]]
# b = []
# print(c + b)
# # print(c + 4)
# a= [3,2,5,2,5,4]





# import collections
# # a = [1,2,2,3,4,5,5]
# # freq = collections.Counter(a)
# # print(freq)


# A = [1,2,3,4]
# B = [3,4,4,6]

# # counter = collections.Counter(a+b for a in A for b in B)

# generator = (a+b for a in A for b in B)

# while True: 
#     print(next(generator))
# # print(counter)

# for g in generator: 
#     print(g)



# a = [1,2]
# a.insert(0, 3)
# print(a)
# a.insert(0, 4)
# print(a)

# d = {"one": 1, "two": 2, "three": 3, "four": 4}

# d['fifth'] = 5
# print(d['fifth'])

# obj = open('afile', 'wb+')
# obj.write('abc')
# obj.read(8)


# """ using with as block no need to mannually close the the file """
# with open('python_notes.txt', 'r+') as f:
#     pass   
    # f_content1 = f.readline()
    # print(f_content1, end='') # get rid of the newline behinded
    # size = 50 
    # chunk = f.read(size)
    # print(chunk, end="")
    
    # while len(chunk) > 0: 
    #     print(chunk, end="")
    #     chunk = f.read(size)#read in 10 char 
    # tell() tells the current positon
    # seek(int) goes to given posion 
    # f.seek(1000)
    # f.write('hahah')
    # f.seek(1000)
    # f.write('p')

# name = input('enter your anme')
# print(name)



# import requests

# res = requests.get('https://api.github.com')
# # print(res)
# # print(res.status_code) 
# # print(res.content)  # raw bytes
# # res.encoding = 'utf-8'
# # serialized = res.text  # serialized JSON content
# # # deserialize 
# # dic = res.json()
# # print(dic)

# # print(res.text) # uTF-8 string

# print(res.headers)  # returns a dictionary-like object
# print(res.headers['Content-Type'])

# s = "  hello world!  "
# """ strip will remove the leading and trailing spaces """
# s = s.strip()
# print(s)

# a = [0 for i in range(3)]
# a = [[0]*3]*3 # this is just created tree references pointed to the same list, if you change one of the element all lists will updated
# a[0][0] = 1

# ar = [[0 for i in range(3)] for j in range(3)]
# print(a)

# for i in range(9, 0, -1):
#     print(i, end='')
# print()

# a = [1,2,3,4]
# b = []
# b.append(a)
# print(b)
# a = a[1:3]
# print(b)

# print(1<<3)

# a = [1,2,3,4]
# a[1:] + a[:1]

# a = 2.1
# print(a % 2)
# print(a//2)

# a = [1,2,3]
# for i in range(len(a))*2:
#     print(a[i])


# stack = [0,1,2]
# ans = [1,2,3]
# ans[stack[-1]] = 5-stack.pop()
# print(ans)

# dic = {1:'a', 2:'b'}
# del dic[1]

# print(dic)

# print(6%3)
# s = 'I'      
# t = ''

# def missingWord(s, t):
#     dic = {}
#     res = []
#     for w in t.split(): 
#         dic[w] = dic.get(w, 0) + 1
#     for w in s.split():
#         if w not in dic or dic[w] == 0: 
#             res.append(w)
#         else: 
#             dic[w] -= 1
#     return res

# # print(missingWord(s, t))

# def no_pair(arr): 
#     res = [0]*len(arr)
#     for j, w in enumerate(arr):
#         n = len(w)
#         count = 1
#         for i in range(1, n+1):
#             if i < n and w[i] == w[i-1]:
#                 count += 1
#             else:
#                 res[j] += (count // 2)
#                 count =  1
#     return res


# # print(no_pair(["ab", "aab", "abb", "abab", "abaaaba"]))
# # print(no_pair(["add", "boook", "break"]))

# def balance(s): 
#     dic = {'(': ')', '[':']', '{':'}'}
#     stack = []
#     for c in s: 
#         if c in dic:
#             stack.append(c)
#         else: 
#             if not stack: return False
#             top = stack.pop()
#             if dic[top] != c: return False
#     return not stack 
        
# # print(balance('([])'))

# def roverMoves(sz, cmd):
#     start = [0, 0]
#     for c in cmd: 
#         if c == 'UP' and start[0] - 1 >= 0: start[0] -= 1
#         if c == 'DOWN' and start[0] + 1 < sz: start[0] += 1
#         if c == 'LEFT' and start[1] - 1 >= 0: start[1] -= 1
#         if c == 'RIGHT' and start[1] + 1 < sz: start[1] += 1
#     return start[0]*sz + start[1]


# # print(roverMoves(4, ["RIGHT", "DOWN", "LEFT", "LEFT", "DOWN"]))
        
# def match_region(g1, g2):
#     n = len(g1)
#     m = len(g1[0])
#     count = 0
#     for i in range(n):
#         for j in range(m):
#             if g1[i][j] == '1' and g2[i][j] == '1' and is_match(g1, g2, i, j): 
#                 count += 1
#     return count

# def is_match(g1, g2, i, j):
#     if i < 0 or i >= len(g1) or j < 0 or j >= len(g1[0]) or g1[i][j] == g2[i][j] == '0': return True
#     if(g1[i][j] == '1' or g2[i][j] == '1') and g1[i][j] != g2[i][j]: return False
#     g1[i][j] = '0'
#     g2[i][j] = '0'

#     return is_match(g1, g2, i-1, j) and is_match(g1, g2, i+1, j) and is_match(g1, g2, i, j-1) and is_match(g1, g2, i, j+1)



# # g1 = [['1', '0', '1'],
# #       ['1', '0', '0'],
# #       ['1', '0', '0']]

# # g2 = [['1', '0', ''],
# #       ['1', '0', '1'],
# #       ['1', '0', '0']]

# # print(match_region(g1, g2))
        
# # hackerrank [4,1,6,8]
# def closest(s, arr):
#     res = []
#     for i in arr:
#         res.append(find(s, i))
#     return res

# def find(s, i):
#     j,k = i-1, i+1
#     left, right = None, None
#     while j >= 0:
#         if s[j] == s[i]: 
#             left = j
#             break
#         j -= 1 
#     while k < len(s):
#         if s[k] == s[i]:
#             right = k
#             break
#         k += 1
#     if not left and not right: return -1
#     if left or right: return left or right
#     return right if i - left > right - i else left



# # print(closest('hackerrank', [4,1,6,8]))
# import collections
# def election(names): 
#     dic = {}
#     for n in names: 
#         dic[n] = dic.get(n, 0) + 1
    
#     res = ''
#     max_count = float('-inf')
#     for k in dic.keys():
#         if dic[k] >= max_count:
#             if dic[k] == max_count and k < res:
#                 continue
#             else:
#                 max_count = dic[k]
#                 res = k
#     return res


# print(election(["Alex", "Michael", "Harry", "Dave","Michael", "Victor", "Harry", "Alex", "Mary", "Mary"]))


# 1. Weird Faculty
# 2. Sub Palindrome
# 3. Twitter new office design
# 4. Twitter Social Network




# dic = {'a': 1}
# del dic['a']
# print(dic)

# a = [1,2]
# b = [1,2]

# print(a == b)


# a = (3,1,2)
# b = (4,1,2)
# print(a < b)


# print(float('inf'))
from collections import defaultdict
city_list = [('TX', 'Austin'), ('TX', 'Houston'), ('NY', 'Albany'), ('NY', 'Syracuse'), ('NY', 'Buffalo'),
             ('NY', 'Rochester'), ('TX', 'Dallas'), ('CA', 'Sacramento'), ('CA', 'Palo Alto'), ('GA', 'Atlanta')]
# cities_by_state = defaultdict(list)
# for state, city in city_list:
#     cities_by_state[state].append(city)
# print(cities_by_state)


cities_by_state = {}
for state, city in city_list:
    cities_by_state[state].append(city)
print(cities_by_state)
