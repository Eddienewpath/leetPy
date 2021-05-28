# 17 think about building the path along and find a terminating condition to collect the path
""" time complexity: O(4^N), for worse case each digit has 4 letters mapping, thus if there are n digits
that is, 4*4*4... that is 4^N
space compelexity:O(n) length of digits deep 
"""
from itertools import product
def letterCombinations(self, digits):
    mapping = ['', '', 'abc', 'def', 'ghi','jkl', 'mno', 'pqrs', 'tuv', 'wxyz']

    def find_comb_path(digits, i, mapping, path, res):
        if len(path) == len(digits):
            res.append(path)
            return
        for j in range(i, len(digits)):
            for c in mapping[int(digits[j])]:
                find_comb_path(digits, j+1, mapping, path+c, res)

    res = []
    if not digits:
            return res 
    find_comb_path(digits, 0, mapping, '', res)
    return res
    
# iterative solution: bfs
import collections
def helper_iter(self, mapping, digits):
    queue = collections.deque()
    queue.append('')
    cur = 0 # index of current processing digit 
    while queue and cur < len(digits):
        size = len(queue)
        while size:
            front = queue.popleft()
            # append front in front of all the neighbors and add to the queue
            for c in mapping[digits[cur]]:
                queue.append(front + c)
            size -= 1
        cur += 1
    return list(queue)



# 39 
# a number can be use unlimited time meaning for each recursive call, the starting position should be current stack frame start position
def combinationSum(self, candidates, target):
    """ be carefull with the python scope and naming if I decide to use outer scope params. """
    def find_comb(start, path, rem):
        if rem == 0:
            res.append(path)
            return

        for i in range(start, len(candidates)):
            """ if the reminder is less than 0, we can just optimize the algo by break out the loop coz we already sort the candidates array, latter candidate
            will be greater than reminder, thus no need to do recursive calls """
            if rem - candidates[i] < 0: break
            find_comb(i, path+[candidates[i]], rem - candidates[i])

    res = []
    candidates.sort()
    find_comb(0,[],target)
    return res

""" dp solution"""
# dp[t]: combinations that sum to t
# dp[t] : dp[t - cand[j]] 
def combinationSum(self, candidates, target):
    candidates.sort()
    res = self.helper(candidates, target)
    return res[target]


def helper(self, cand, tar):
    dp = [[] for _ in range(tar+1)]
    # for every t: 
    for t in range(tar+1):
        tmp = []
        j = 0
        while j < len(cand) and cand[j] <= t: 
            if cand[j] == t:
                tmp.append([cand[j]])
            else: 
                for comb in dp[t-cand[j]]:
                    # for every cand[j] < t, we start from j == 0, so there could be duplicates
                    # to avoid duplicates [2,2,3] [2,3,2], we mantain the combination list in ascending order
                    if cand[j] >= comb[-1]:
                        tmp.append(comb + [cand[j]])
            j += 1
        dp[t] = tmp
    return dp
            

# 40 
def combinationSum2(self, candidates, target):
    def find_comb(start, path, rem):
            if rem == 0:
                res.append(path)
                return
            for i in range(start, len(candidates)):
                """ be careful with the i-1>= start condition, don't code it like i-1>= 0
                otherwise you will skip some case such as 1,1,6
                on the other hand, if there are duplicates, we need to skip the latter ones
                 """
                if i-1 >= start and candidates[i] == candidates[i-1]: continue
                if rem - candidates[i] < 0: break
                find_comb(i+1, path+[candidates[i]], rem - candidates[i])

    res = []
    candidates.sort()
    find_comb(0, [], target)
    return res



""" dp solution """
# using tuple becuase list() must be hashable/inmutable
""" Do not understand """
def combinationSum2(self, candidates, target):
    candidates.sort()
    dp = [set() for i in range(target+1)]
    dp[0].add(())
    for num in candidates:
        for t in range(target, num-1, -1):
            for prev in dp[t-num]:
                dp[t].add(prev + (num,))
    return list(dp[-1])

""" Do not understand """

# 77
def combine(self, n, k):
    def build_comb(start, path, rem):
        if rem == 0:
            res.append(path)
            return
        for i in range(start, n+1):
            build_comb(i+1, path+[i], rem-1)

    res = []
    build_comb(1, [], k)
    return res
                

""" iterative using pointer """
# making all the elements in the path increment by 1 initially,and then working backward from the position k-1 backed to 0
def combine_iter(self, n, k):
    res, path = [], [0]*k
    i = 0
    while i >= 0:
        path[i] += 1
        # moving backward if current position is exceeding the n
        if path[i] > n:
            i -= 1
        # add the result
        elif i == k-1:
            res.append(path[:])
        # moving forward if path[i] <= n and i < k-1
        else:
            i += 1
            # this line and the "path[i] += 1" line can guruntee [i] > [i-1] 
            path[i] = path[i-1]
    return res


""" iterative dfs """ 
def combine(self, n, k):
    ans, stack, x = [], [], 1
    while stack or x <= n:
        while len(stack) < k and x <= n:
            stack.append(x)
            x += 1

        if len(stack) == k:
            ans.append(stack[:])

        x = stack.pop() + 1
    return ans


""" binary sorted  """
""" Do not understand """
def combine(self, n, k):
    # init first combination
    nums = list(range(1, k + 1)) + [n + 1]

    output, j = [], 0
    while j < k:
        # add current combination
        print(nums)
        output.append(nums[:k])
        # increase first nums[j] by one
        # if nums[j] + 1 != nums[j + 1]
        j = 0
        while j < k and nums[j + 1] == nums[j] + 1:
            nums[j] = j + 1
            j += 1
        nums[j] += 1

    return output
""" Do not understand """

""" implement math solution """
# formula C(n, k) = C(n-1, k-1) + C(n-1, k), very easy to prove
# means if n is selected, go select k-1 from n-1, elif n is not selectd, go select k from n-1
def combine(self, n, k):
        # C(n-1, k) 
        if n == k:
            return [[i for i in range(1, n+1)]]
        # C(n-1, k-1)
        if k == 1:
            return [[i] for i in range(1, n+1)]

        return self.combine(n-1, k) + [[n] + j for j in self.combine(n-1, k-1)]


""" implement dp solution """
# C(n, k) = C(n-1, k-1) + C(n-1, k)
def combine(self, n, k):
    C = [[[] for _ in range(k+1)] for _ in range(n+1)]
    for r in range(n+1):
        C[r][0] = [[]]

    for i in range(1, n+1):
        for j in range(1, k+1):
            if j > i:
                break
            if i-1 >= j:
                C[i][j] += C[i-1][j]
            C[i][j] += [c + [i] for c in C[i-1][j-1]]

    return C[n][k]


# 78
def subsets(self, nums):
    def build_subsets(start, path):
        # similar to preorder
        res.append(path)
        for i in range(start, len(nums)):
            build_subsets(i+1, path+[nums[i]])

    res = []
    build_subsets(0,[])
    return res

""" iterative solution """
def subsets(self, nums):
    res = [[]]
    for n in nums:
        size = len(res)
        for i in range(size):
            res.append(res[i] + [n])
    return res

# same idea but pythonic way
def subsets(self, nums):
    res = [[]]
    for num in sorted(nums):
        res += [item+[num] for item in res]
    return res
""" example: [1,2,3]
[]
[] -> [1]
[] [1] -> [2] [1,2]
[] [1] [2] [1,2] -> [3] [1,3] [2,3] [1,2,3] 
"""


""" bit manipulation solution """
def subsets(self, nums):
    n = len(nums)
    res = []
    for i in range(2**n, 2**(n+1)):
        # 2**(n+1) - 1 can get 11..11
        # generate 00..0 to 11..1 
        # bin() function return a string of binary representation starts with string '0b'
        bitmask = bin(i)[3:]
        tmp = []
        for j in range(n):
            if bitmask[j] == '1':
                tmp.append(nums[j])
        res.append(tmp)
    return res


# pythonic way and also faster
def subsets(self, nums):
    n = len(nums)
    res = []
    for i in range(2**n, 2**(n+1)):
        bitmask = bin(i)[3:]
        res.append([nums[j] for j in range(n) if bitmask[j] == '1'])
    return res



# 90
def subsetsWithDup(self, nums):
    def build_comb(start, path):
        res.append(path)
        for i in range(start, len(nums)):
            if i - 1 >= start and nums[i] == nums[i-1]: continue
            build_comb(i+1, path+[nums[i]])

    res = []
    nums.sort()
    build_comb(0, [])
    return res


""" iterative solution """
# using set but slow
def subsetsWithDup(self, nums):
    res = [[]]
    nums.sort()
    for n in nums:
        size = len(res)
        for i in range(size):
            res.append(res[i] + [n])

    ans = set([tuple(l) for l in res])
    return list(ans)


# without using set
def subsetsWithDup(self, nums):
    res = [[]]
    nums.sort()
    size = 0
    for i in range(len(nums)):
        k = size if i-1 >= 0 and nums[i] == nums[i-1] else 0
        size = len(res)
        res += [res[j] + [nums[i]] for j in range(k, size)]
    return res


# readable version
def subsetsWithDup(self, nums):
    res = [[]]
    nums.sort()
    size = 0
    for i in range(len(nums)):
        k = size if i-1 >= 0 and nums[i] == nums[i-1] else 0
        # current res size and update to outer scope 
        size = len(res)
        # then add new comb to the res
        # if dup exists, we only add nums[i] to comb from [k, size]
        for j in range(k, size):
            res.append(res[j] + nums[i])
    return res


""" bit manipulation """
# hard to understand
# https: // leetcode.com/problems/subsets-ii/discuss/30325/Java-solution-using-bit-manipulation



# 216 
def combinationSum3(self, k, n):
    def build_comb(start, path, k, n):
        if k == 0 and n == 0: 
            res.append(path)
            return 
        
        for i in range(start, 10):
            if n - i < 0: break
            build_comb(i+1, path+[i], k-1, n-i)
        
    res = []
    build_comb(1, [], k, n)
    return res


# 46
def permute(self, nums):
    def get_permute(nums, path):
        if len(nums) == 0:
            res.append(path)
            return

        for i in range(len(nums)):
            get_permute(nums[:i]+nums[i+1:], path+[nums[i]])

    res = []
    get_permute(nums, [])
    return res


""" iterative solution """
""" 
python slice will not throw error for example a = '1', a[:5] = '1' and a[5:] = '' 
"""
def permute(nums):
    perms = [[]]
    # insert each n into result perms 
    for n in nums:
        tmp = []
        # for every perm in perms, insert current n into every possible insertion point. 
        for p in perms:
            for i in range(len(p)+1):
                tmp.append(p[:i] + [n] + p[i:])
        # update perms list
        perms = tmp
    return perms



# 47 
def permuteUnique(self, nums):
    def get_permute(nums, path):
        if len(nums) == 0:
            res.append(path)
            return
        for i in range(len(nums)):
            if i-1 >= 0 and nums[i] == nums[i-1]:
                continue
            get_permute(nums[:i]+nums[i+1:], path+[nums[i]])
    nums.sort()
    res = []
    get_permute(nums, [])
    return res


""" iterative """
def permuteUnique(self, nums):
    perms = [[]]
    for n in nums:
        tmp = []
        for p in perms:
            # inserting right before i-th position
            for i in range(len(p)+1):
                tmp.append(p[:i] + [n] + p[i:])
                # after insert before i-th position, not on the original list, check if at i-th position is same as the current 
                # n, if the same we can terminate because insert after i will be the same as insert before. 
                # i < len(p) because there is nothing after i == len(p)
                if i < len(p) and p[i] == n: break
        perms = tmp 
    return perms



# 784 
def letterCasePermutation(self, S):
    def build_path(i, path):
        if len(path) == len(S):
            res.append(path)
            return
        """ for every position, add swapped case characters to previous path or add original case to previous path or non character to the previous path"""
        if S[i].isalpha():
            build_path(i+1, path+S[i].swapcase()) # string concatenation will create a new string path thus below concatenation will create a new one
        build_path(i+1, path+S[i])
    res = []
    build_path(0, '')
    return res


""" over recursive solution, very slow """
def dfs(self, start, s, path, res):
    if path and len(path) == len(s):
        res.append(path)
        return
    # this means that the path starts with char in [0, len(s)], but it is unnecessary because the final path will be the same length of the orginal one 
    # if we start with second or later char, it will destine to be wrong path. 
    # in this case we only have two states, swap or keep it as it is. 
    for i in range(start, len(s)):
        if s[i].isalpha():
            self.dfs(i+1, s, path + s[i].swapcase(), res)
        self.dfs(i+1, s, path + s[i], res)


""" implement bfs solution """
def letterCasePermutation(self, S):
    queue = collections.deque()
    queue.append(S)
    # every char has two states need to be add the queue, if not char, skip that position 
    for i in range(len(S)):
        size = len(queue)
        if S[i].isdigit():
            continue
        while size:
            front = queue.popleft()
            queue.append(front)
            queue.append(front[:i] + front[i].swapcase() + front[i+1:])
            size -= 1
    return list(queue)


""" best and clean and pythonic solution """
def letterCasePermutation(self, S):
    res = ['']
    for c in S:
        if c.isalpha():
            res = [i + j for i in res for j in (c.lower(), c.upper())]
        else:
            res = [i + c for i in res]
    return res


""" python lib """
# itertools.product(alist of iterables) is equal nested loops or cartitian product of two iterable elements
# for example: product([1,2], [3,4])  => [(1,3), (1,4), (2,3), (2,4)] 
# *L here meaning more than two iterables need to find cartitian products
def letterCasePermutation(S):
    from itertools import product
    L = [[i.lower(), i.upper()] if i.isalpha() else i for i in S]
    return [''.join(i) for i in product(*L)]



#79
def exist_old(self, board, word):
    def has_path(path, i, j, w):
        if not w:
            return True
        if i < 0 or i >= len(board) or j < 0 or j >= len(board[0]) or board[i][j] == '#':
            return False

        tmp = board[i][j]
        board[i][j] = '#'
        # be careful with the directions initializing
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        for x, y in directions:
            """ check equality char by char along the way """
            if tmp == w[0] and has_path(path+tmp, i+x, j+y, w[1:]):
                return True
        # if four directions all falses then we need to change back to original element so that next position can be re using this element
        board[i][j] = tmp
        return False

    n, m = len(board), len(board[0])
    for i in range(n):
        for j in range(m):
            if board[i][j] == word[0] and has_path('', i, j, word):
                return True
    return False


""" cleaner implementation dfs  """
def exist(self, board, word):
        n, m = len(board), len(board[0])
        for i in range(n):
            for j in range(m):
                if self.dfs(board, i, j, word):
                    return True
        return False


def dfs(self, b, i, j, w):
    if not w: return True
    if i < 0 or i >= len(b) or j < 0 or j >= len(b[0]) or b[i][j] != w[0]: 
        return False

    old = b[i][j] 
    b[i][j] = '#'
    for x , y in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
        if self.dfs(b, i+x, j+y, w[1:]):
            return True
    b[i][j] = old



# 22 
# choices: open and closing parenthesis 
# constraints: 
# 1. open parenthesis and closing parenthesis must less and equal to n
# 2. valid pair of parenthesis must start with open, thus the open cannot greater than closing parenthesis, thus op <= cl meaning the open paren available must less than or equal to closing paren available
def generateParenthesis(self, n):
    def build_paren(op, cl, path, res):
        if op == 0 and cl == 0: 
            res.append(path)
            return
        # constraints 
        if op-1 >= 0 and op <= cl:
            build_paren(op-1, cl, path+'(', res)
        if cl-1 >= 0 and op <= cl:
            build_paren(op, cl-1, path+')', res)
            
    res = []
    build_paren(n, n, '', res)
    return res



""" cleaner version """
def dfs(self, left, right, path, res):
    if not left and not right:
        res.append(path)
        return
    #invariant
    # if left == right, must reduce the closing, if left < right, you can reduce either
    # below conditon, cannot change order
    if left <= right: 
        if left > 0:
            self.dfs(left-1, right, path + '(', res)

        if right > 0: 
            self.dfs(left, right-1, path + ')', res)



""" dp solution """
""" Do not understand """


def generateParenthesis(self, n):
    dp = [[] for i in range(n + 1)]
    dp[0].append('')
    for i in range(n + 1):
        for j in range(i):
            dp[i] += ['(' + x + ')' + y for x in dp[j]
                        for y in dp[i - j - 1]]
    return dp[n]

""" Do not understand """
