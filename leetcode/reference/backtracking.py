""" goal: 24 problems """
# each stack frame has its state and the thing common to all stack frame is they all follow same constraint. 

# p22
# pattern for doing backtracking problems, think about three points: 
# 1. what are the choices/options, for this problem are the open and closing parenthesis
# 2. what are the constraints, for this problem is the number of open should equal or less than the closing parenthesises. 
# 3. what is the goal, for this problem the goal is the place n*2 total open and closing parenthesises to form valid parenthesis combination

def generateParenthesis(self, n: int) -> List[str]:
    def genParen(n, m, path, res):
        if n == 0 and m == 0:
            res.append(path)
            return
        # (   
        if n <= m and n-1 >= 0:
            genParen(n-1, m, path + '(', res)

        # )
        if n <= m and m-1 >= 0:
            genParen(n, m-1, path + ')', res)

    res = []
    genParen(n, n, '', res)
    return res


# p17 
# choices: each digit corresponding to a list of letters 
# constraints: each digit you can only choose one char, must choose according to the order of the digits 
# goal: all combinations of given digits can represent 
# base case: if the path is equal to the length of the digits, append the result
def letterCombinations(self, digits: str) -> List[str]:
    if not digits: return []
    arr = ['', '', 'abc', 'def', 'ghi','jkl', 'mno', 'pqrs', 'tuv', 'wxyz']
    n = len(digits)

    def genComb(digits, arr, path, res):
        if len(path) == n:
            res.append(path)
            return
        for i, d in enumerate(digits):
            for c in arr[int(d)]:
                genComb(digits[i+1:], arr, path + c, res)

    res = []
    genComb(digits, arr, '', res)
    return res


# p46
# choices: starts with different int
# constraints: can use each number once for a permutation, 
# get all permutations of given list of numbers
def permute(self, nums: List[int]) -> List[List[int]]:
    def get_permute(nums, path, res, n):
        if len(path) == n:
            dup = path[::]
            res.append(dup)
            return
        for i in range(len(nums)):
            path.append(nums[i])
            get_permute(nums[:i] + nums[i+1:], path, res, n)
            # because we are operate on the same list, unlike string, when we do concatenation we created a copy of the string
            path.pop()

    if not nums: return []
    res , n = [], len(nums)
    get_permute(nums, [], res, n)
    return res


# p47 
# choices: int in list 
# constraint: do not include dup
# goal:get all permute 
# base: 
def permuteUnique(self, nums):
    def get_permute(nums, path, res):
        if not nums: 
            res.append(path)
            return
        for i in range(len(nums)):
            if i+1 < len(nums) and nums[i+1] == nums[i]: continue
            get_permute(nums[:i]+nums[i+1:], path+[nums[i]], res)
    res = []
    nums.sort()
    get_permute(nums, [] , res)
    return res 



# p60 
# TLE
def getPermutation(self, n, k):
    def get_k_permut(n, p, res):
        if not n:
            res.append(p)
            return
        for i in range(len(n)):
            get_k_permut(n[:i] + n[i+1:], p + str(n[i]), res)

    res = []
    get_k_permut(range(1, n+1), '', res)
    return res[k-1]





# 39
# choices for a stack frame: one of the candidates, note that candidates can be selected unlimited times 
# constraint: can not choose previous candiate, can choose current on unlimited times 
# goal: add all the possible sets of candidates that can sum to target to the result list
# base case: if target == 0: add the result

# tip: be careful with the scope of python, you could end up using outer scope variable in the inner function. 
# in this case, name your comb_sum param cand instead of candidates coz if you dont, you will end up using the candidates out side. 
# that will cause bug. 

def combinationSum(self, candidates, target):
    def comb_sum(cand, tmp, res, tar): 
        if tar < 0: return 
        if tar == 0: 
            dup = tmp[::]
            res.append(dup)
            return 
        for i in range(len(cand)):
            tmp.append(cand[i])
            comb_sum(cand[i:], tmp, res, tar - cand[i])
            tmp.pop()
    
    if not candidates: return []
    res = []
    comb_sum(candidates, [], res, target)
    return res 


# p216 
# choice: [0-9], use once for each set 
# constraint: can use same number once in a set
def combinationSum3(self, k, n):
    def comb_sum(k, t, start, path, res):
        if k == 0 and t == 0: 
            res.append(path)
            return

        for i in range(start, 10):
            comb_sum(k-1, t - i, i+1, path + [i], res)

    res = []
    comb_sum(k,n, 1, [], res)
    return res


# p79 
# choices: four directions neibours
# constraint: cannot choose same cell more than once
# goal: search given word 
# base case: reach the boundry or found the word   
# 
# use path to do string concatenation will LTE 
def exist(self, board, word):
    def dfs(b, i, j, w):
        if len(w) == 0: return True
        if i < 0 or j < 0 or i >= len(b) or j >= len(b[0]) or b[i][j] == '*':
            return False

        directions = [(-1, 0), (1, 0), (0, 1), (0, -1)]
        for x, y in directions:
            cur = b[i][j]
            b[i][j] = '*'
            if cur == w[0] and dfs(b, i+x, j+y, w[1:]): return True
            b[i][j] = cur # dont forget to change back, coz this is backtracking
        return False

    if not board or not word: return False
    n, m = len(board), len(board[0])
    for i in range(n):
        for j in range(m):
            if board[i][j] == word[0] and dfs(board, i, j, word):
                return True 
    return False
        

# p78
# intuition: because the longer subset is actually formed by the small subset 
# choices: for subset starts with [i], the choices are in [i+1:] 
# constraint: 
# goal: place element into given length of subset array
# base case: when all the elements are starting the subset. 
# python tip: [] + [] will create a new list, this can get rid of dup = path[::] like previous problems

# top down build the path along 
def subsets(self, nums):
    def backtrack(nums, path, res):
        res.append(path)
        for i in range(len(nums)):
            # use previous path to form current path
            backtrack(nums[i+1:], path+[nums[i]], res)
    res = []
    backtrack(nums, [], res)
    return res

# p90 
def subsetsWithDup(self, nums):
    def backtrack(nums, path, res):
        res.append(path)
        for i in range(len(nums)):
            # skip the latter dup coz current copy will need to use latter dup to make subset.  if you skip previous dup, there will be subsets missing
            if i-1 >= 0 and nums[i] == nums[i-1]: continue
            backtrack(nums[i+1:], path + [nums[i]], res)
        
    res = []
    nums.sort()
    backtrack(nums, [], res)
    return res 


# p93
# choices: three choices, length 1, 2, 3, for a block of number 
# constraints: number needs to be in the range [0, 255], and for number length longer than 1, first digit cannot be 0
# goal: generate all valid ip 
# base: idx == 4 and s is empty meaning all four section is separated
def restoreIpAddresses(self, s):
    def ip_generator(s, idx, path, res):
        # base 
        if idx == 4 and not s: 
            res.append(path[:-1])
            return 
        # 3 choices
        for i in range(1, 4):
            # constraint
            # the s should never less than i 
            if i <= len(s):
                if i == 1: 
                    ip_generator(s[i:], idx+1, path + s[:i] + '.', res)
                if i == 2 and s[0] != '0': 
                    ip_generator(s[i:], idx+1, path + s[:i] + '.', res)
                if i == 3 and int(s[:i]) <= 255 and s[0] != '0':
                    ip_generator(s[i:], idx+1, path + s[:i] + '.', res)
    
    if len(s) > 12: return []
    res = []
    ip_generator(s, 0, '', res)
    return res 


# p131 
# choices: every char in the s is a possibe partition position
# constraint: the substring need to be a palindrome
# goal: find all partions to partion the s into substrings which are palindromes 
# base:
def partition(self, s):
    def backtrack(s, path, res):
        if not s:
            res.append(path)
            return
        # choices 
        for i in range(len(s)):
            # constraint
            if s[:i+1] == s[:i+1][::-1]:
                backtrack(s[i+1:], path + [s[:i+1]], res)
    res = []
    backtrack(s, [], res)
    return res 


# p77
# choices: [1, n]
# constaint: k slots
# goal: find all combinations with length of k.
# base: when k == 0, append result
def combine(self, n, k):
    def comb_k(start, n, k, path, res):
        if k == 0:
            res.append(path)
            return 
        for i in range(start, n+1):
            comb_k(i+1, n, k-1, path+[i], res)
    res = []
    comb_k(1, n, k, [], res)
    return res 



# p784 
# choice: digit skip, (low letter or high letter)
# constraint: if it is digit just add to the path
# goal: convert given string into differnt string by making lower letter to higher letter, or vice versa. 
# base: if the length of path is equal to the original string, append the answer.
# tricky part is the string could end with a digit
# pass the cases, but very slow, meant for practicing backtracking.
def letterCasePermutation(self, S):
    def backtrack(s, path, res, n):
        if not s and len(path) == n:
            res.append(path)
            return

        for i in range(len(s)):
            if s[i].isdigit():
                path += s[i]
            else:
                backtrack(s[i+1:], path + s[i].lower(), res, n)
                backtrack(s[i+1:], path + s[i].upper(), res, n)

        if s and s[-1].isdigit() and len(path) == n:
            res.append(path)
            return

    if not S: return []
    res = []
    backtrack(S, '', res, len(S))
    return res


# p526
# choice: [i:N] for in [1, N]
# constarint: (index + 1) % [idx] or [idx] % (index + 1) is 0 
# goal: find all permutation st fullfils the constraint
# base: when array is empty 
# tricky part: the index of the original should be pass along the recursion
def countArrangement(self, N):
    def construct(nums, start, path, res):
        if not nums and len(path) == N:
            res.append(path)
            return
        
        for i in range(len(nums)):
            if start % nums[i] == 0 or nums[i] % start == 0: 
                construct(nums[ :i] + nums[i+1: ], start + 1, path + [nums[i]], res)  

    res = []
    construct(range(1, N+1), 1, [], res)
    return len(res)


# p1079
# 1.find all the subsets
# 2. find the permutation of all the subsets

# there is a shorter answer, here just meant for practcing backtracking 
def numTilePossibilities(self, tiles):
    def findsubsets(s, path, res):
        res.append(path)
        for i in range(len(s)):
            if i-1 >= 0 and s[i] == s[i-1]: continue
            findsubsets(s[i+1: ], path + s[i], res)

    def findpermute(s, path, res):
        if not s:
            res.append(path)
            return 

        for i in range(len(s)):
            if i-1 >= 0 and s[i] == s[i-1]: continue
            findpermute(s[:i] + s[i+1: ], path + s[i], res)
        
        if not tiles: return []
        tiles = ''.join(sorted(list(tiles)))
        subsets, res = [], []
        findsubsets(tiles, [], subsets)
        for s in subsets:
            if len(s) == 1: 
                res.append(s)
            else:
                findpermute(s, [], res)
        return len(res)

        
# p320
# choices: for each char we have two choices: keep or abbreviate. 
# constraint: 
# goal: construct all valid abbreviations 
# base: 
def generateAbbreviations(self, word):
    def get_abbr(w, i, path, res, cnt):
        if i == len(w):
            if cnt > 0:
                path += str(cnt)
            res.append(path)
            return

        # abbr
        get_abbr(w, i+1, path, res, cnt+1)
        # keep
        get_abbr(w, i+1, path + (str(cnt) if cnt > 0 else '') + w[i], res, 0)

    res = []
    get_abbr(word, 0, '', res, 0)
    return res


# p37 
# choices: each cell choose from [1, 9]
# constraint: each number should appear in a row and col once. 
# goal: fill the cell
# base:
def solveSudoku(self, board):
    def backtrack(board):
        n = len(board)
        for i in range(n):
            for j in range(n):
                if board[i][j] == '.':
                    for c in ['1', '2', '3', '4', '5', '6', '7', '8' ,'9']: 
                        if valid_cell(i, j, c, board):
                            board[i][j] = c
                            if backtrack(board): 
                                return True
                            else:
                                board[i][j] = '.'
                    return False
        return True

    def valid_cell(i, j, ch, b):
        for k in range(9):
            # check row
            if b[i][k] != '.' and b[i][k] == ch: return False
            # check col 
            if b[k][j] != '.' and b[k][j] == ch: return False
            # check block 3*3 
            if b[3 * (i // 3) + k // 3][3 * (j // 3) + k % 3] != '.' and b[3 * (i // 3) + k // 3][3 * (j // 3) + k % 3] == ch:
                return False      
        return True

    if not board: return
    backtrack(board)
            

""""""""""""""""""""""""""""""""
""" Backtracking """


def permute(nums):
    res = []

    def backtrack(nums, path):
        if not nums:
            res.append(path)
        for i in range(len(nums)):
            # building a path starts with [i], and backtracking the rest of elements
            backtrack(nums[:i]+nums[i+1:], path+[nums[i]])
    backtrack(nums, [])
    return res


def permuteUnique(nums):
    res = []

    def backtrack(nums, path):
        if not nums:
            res.append(path)
        for i in range(len(nums)):
            if i+1 < len(nums) and nums[i] == nums[i+1]:
                continue
            backtrack(nums[:i]+nums[i+1:], path+[nums[i]])
    nums.sort()
    backtrack(nums, [])
    return res


# bottom -> up
def subsets_bottom_up(nums):
    if not nums:
        return [[]]
    for i in range(len(nums)):
        res = subsets_bottom_up(nums[i+1:])
        res += [[nums[i]] + l for l in res]
        return res

# dfs optimized


def subsets_top_down(nums):
    def build_path(nums, start, path, res):
        res.append(path)
        for i in range(start, len(nums)):
            build_path(nums, i+1, path+[nums[i]], res)
    res = []
    build_path(nums, 0, [], res)
    return res


def subsetsWithDup(nums):
    def build_path_skip_dup(nums, start, path, res):
        res.append(path)
        for i in range(start, len(nums)):
            if i-1 >= start and nums[i] == nums[i-1]:
                continue  # skip the later dup
            build_path_skip_dup(nums, i+1, path+[nums[i]], res)
    nums.sort()
    res = []
    build_path_skip_dup(nums, 0, [], res)
    return res


def combinationSum(candidates, target):
    def dfs(candidates, target, path, start, res):
        if target < 0:
            return
        if target == 0:
            res.append(path)
            return
        for i in range(start, len(candidates)):
            dfs(candidates, target-candidates[i], path+[candidates[i]], i, res)
    res = []
    dfs(candidates, target, [], 0, res)
    return res


def combinationSum2(candidates, target):
    def dfs(candidates, target, path, start, res):
        if target < 0:
            return
        if target == 0:
            res.append(path)
            return
        for i in range(start, len(candidates)):
            if i-1 >= start and candidates[i] == candidates[i-1]:
                continue
            dfs(candidates, target -
                candidates[i], path+[candidates[i]], i+1, res)
    candidates.sort()
    res = []
    dfs(candidates, target, [], 0, res)
    return res


def combinationSum3(k, n):
    def dfs(k, n, arr, path, res, start):
        if k < 0 or n < 0:
            return
        if k == 0 and n == 0:
            res.append(path)
            return
        for i in range(start, len(arr)):
            dfs(k-1, n-arr[i], arr, path+[arr[i]], res, i+1)

    arr = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    res = []
    dfs(k, n, arr, [], res, 0)
    return res
