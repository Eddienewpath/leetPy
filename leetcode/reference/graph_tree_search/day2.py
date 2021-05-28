# 542 
# naive solution doing bfs for every 1's 
def updateMatrix(self, matrix):
    n, m = len(matrix), len(matrix[0])
    res = [[None for _ in range(m)] for _ in range(n)]

    for i in range(n):
        for j in range(m):
            if matrix[i][j] == 1:
                res[i][j] = self.bfs(matrix, i, j)
            else:
                res[i][j] = 0
    return res

import collections
def bfs(self, m, i, j):
    queue = collections.deque()
    queue.append((i, j))
    steps = 0 
    while queue:
        size = len(queue)
        while size:
            i, j = queue.popleft()
            if m[i][j] == 0: return steps
            for x, y in [(1,0), (-1, 0), (0, 1), (0, -1)]:
                if i+x >= 0 and j + y >= 0 and i + x < len(m) and j+y < len(m[0]):
                    queue.append((i+x, j+y))
            size -= 1
        steps += 1
    return -1
                
            
""" dont understand """ 
""" instead of finding shortest path from 1 to 0, the trick is do bfs on every non 1 cell and accumulate from bottom up """
def updateMatrix(self, matrix):
    n, m = len(matrix), len(matrix[0])
    import collections
    queue = collections.deque()
    for i in range(n):
        for j in range(m):
            if matrix[i][j] == 0:
                queue.append((i, j))
            else:
                matrix[i][j] = float('inf')

    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    while queue:
        i, j = queue.popleft()
        for x, y in directions:
            r = i + x
            c = j + y
            z = matrix[i][j] + 1

            if r < 0 or r >= n or c < 0 or c >= m or matrix[r][c] <= z:
                    continue

            queue.append((r, c))
            matrix[r][c] = z
    return matrix


""" implement dp solution """




# 93
def restoreIpAddresses(self, s):
    def build_ip(s, k, path):
        if k == 0:
            # when k is 0, there could be some digits left behind, so if nothing left we have a valid ip
            if not s:
                path = path[:-1]
                res.append(path)
                return

        # each stack frame we have 3 choices, choose 1, 2, 3 digits
        for i in range(1, 4):
            # the substring s should always at least i long, this condition easy to forget.
            if i <= len(s):
                if i == 1:
                    build_ip(s[i:], k-1, path+s[:i]+'.')
                # if choose 2 or 3 digits, the string cannot start with 0
                elif i == 2 and s[0] != '0':
                    build_ip(s[i:], k-1, path+s[:i]+'.')
                elif i == 3 and s[0] != '0' and int(s[:i]) <= 255:
                    build_ip(s[i:], k-1, path+s[:i]+'.')

    res = []
    if len(s) > 12:
        return res
    build_ip(s, 4, '')
    return res

# without modify s version


def restoreIpAddresses(self, s: str) -> List[str]:
    res = []
    self.build(0, s, '', 4, res)
    return res


def build(self, start, s, path, n, res):
    if start > len(s) or n < 0: return 
    if start == len(s):
        if n == 0: 
            res.append(path[:-1])
        return
    
    for i in range(1, 4):
        cur = s[start:start+i]
        if i == 1:
            self.build(start + 1, s, path + cur + '.', n-1, res)
        elif i == 2 and cur[0] != '0':
            self.build(start + 2, s, path + cur + '.', n-1, res)
        elif i == 3 and cur[0] != '0' and int(cur) <= 255:
            self.build(start + 3, s, path + cur + '.', n-1, res)

# 131
""" things to notice
for example: a = [1,2,3] if a + [4] this will return a new array [1,2,3,4], a will stay the same [1,2,3]
this is why if you do a using + operator, you dont need to pop() and [:] operation to do backtracking 
unlike other programming languages, java only maintain one array thru out the algorithm
 """
def partition_old(self, s):
    def build_comb(s, start, path):
        if start == len(s):
            res.append(path)
            return

        for i in range(start, len(s)):
            if s[start:i+1] == s[start:i+1][::-1]:
                build_comb(s, i+1, path+[s[start:i+1]])

    res = []
    if not s:
        return res
    build_comb(s, 0, [])
    return res


""" cleaner version """

def partition(self, s):
    res = []
    self.dfs(s, [], res)
    return res

def dfs(self, s, path, res):
    if not s and path: 
        res.append(path)
        return 
    
    for i in range(1, len(s)+1):
        if s[:i] == s[:i][::-1]:
            self.dfs(s[i:], path + [s[:i]], res)
    
 
# solution that maintain only one temp array and the end copy all the pal and add to result 
def partition(self, s):
    def build_comb(s, start, path):
        if start == len(s):
            tmp = path[:]
            res.append(tmp)
            return 
        
        for i in range(start, len(s)):
            if s[start:i+1] == s[start:i+1][::-1]:
                path.append(s[start:i+1])
                build_comb(s, i+1, path)
                path.pop()
                
    res = []
    if not s: return res
    build_comb(s, 0, [])
    return res


""" DP solution """




# 698
""" two ways to consider what choices are for each stack frame
1. for each element, there are k buckets to put, then the goal is to use out all the elements to fill all the buckets
2. for each bucket, there is a list of items to select, then the goal is fill each buckets with items in the list
"""
# choices: k same fix-sized buckets for every items in nums, to put element one by one
# constraints: each bucket must have enough space to fill the number
# goal: place each number into k buckets

def canPartitionKSubsets_sorted(self, nums, k):
    if len(nums) < k: return False 
    s = sum(nums)
    # naturally you put the bigger object into the buckets first, this will give you more flexibility. 
    nums.sort(reverse=True) 
    if s % k != 0: return False
    # create k equal size buckets
    sub = s//k
    buckets = [sub]*k
    
    # pos is current working element position
    def fill_buckets(pos):
        if pos == len(nums): 
            return True
        # go thru each bucket find the bucket that can fill the element at pos and successfully fill the rest of buckets 
        # otherwise, you cannot put pos element at current bucket so you need to take it out. 
        for i in range(k):
            if buckets[i] >= nums[pos]:
                buckets[i] -= nums[pos]
                if fill_buckets(pos+1):
                    return True
                buckets[i] += nums[pos]
                # because all the buckets are same size, if current bucket become empty after searching, this means there is not way
                # all the buckets can hold current working element. return false early
                if buckets[i] == sub:
                     break
        return False
    return fill_buckets(0)


# choices: for each bucket, choice is nums , to fill bucket one by one, 
# constraints: each bucket must have enough space to fill the number
# goal: fill each bucket 
def canPartitionKSubsets(self, nums, k):
    if len(nums) < k: return False 
    s = sum(nums)
    if s % k != 0: return False
    target = s//k
    # to mark visted elements
    visited = [False for _ in range(len(nums))]

    # num_cnt: denotes number of elemnets inside current bucket
    def dfs(num_cnt, start, k, total):
        if k == 1: return True
        # if total meets the target and number of elemnts inside current buckets more than 0, we can move on to rest k-1 buckets
        # reset start index back to 0 coz some of the elements may not be visited previously
        if total == target and num_cnt > 0: 
            return dfs(0, 0, k-1, 0)
        # for current stack, we have choices from [start, len(nums)]
        for i in range(start, len(nums)):
            if not visited[i]:
                visited[i] = True
                # if current element is not visted, add to the total and increment the start index and num_cnt 
                if dfs(num_cnt+1, i+1, k, total+nums[i]): return True
                visited[i] = False
        return False
    
    return dfs(0, 0, k, 0)
                    
                    
                
""" DP solution """