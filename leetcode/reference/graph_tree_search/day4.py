# 301 
# TLE 
class Solution:
    def removeInvalidParentheses(self, s):
        if not s:
            return ['']
        l, r = self.invalid_count(s)
        res = []
        self.dfs(s, l, r, res)
        return res

    # """ smart way to count unbalanced open and closing paren """
    def invalid_count(self, s):
        l, r = 0, 0
        for c in s:
            l += (c == '(')
            """ when opening paren is 0 and current paren is closing, then the unbalanced closing should increment by 1 
            if opening unbalanced paren is not 0, and current paren is closing paren, then meaning they are matched, so decrese openning 
            unbalanced paren count by 1 
            """ 
            if l == 0:
                r += (c == ')')
            else:
                l -= (c == ')')
        return [l, r]

    """ simple way to check if paren are matched for not, we do not need to use stack to do it when there is only one type of paren  """
    def is_valid(self, s):
        count = 0
        for c in s:
            if c == '(':
                count += 1
            elif c == ')':
                count -= 1
            # whenever count is less than 0 during the loop, meaning there is unbalanced closing paren
            if count < 0:
                return False
        return count == 0

    def dfs(self, s, l, r, res):
        if l == 0 and r == 0:
            if self.is_valid(s) and s not in res:
                res.append(s)
            return

        for i in range(len(s)):
            if i > 0 and s[i] == s[i-1]:
                continue
            if s[i] in '()':
                if r > 0:
                    self.dfs(s[:i]+s[i+1:], l, r-1, res)
                elif l > 0:
                    self.dfs(s[:i]+s[i+1:], l-1, r, res)


obj = Solution()
res = obj.removeInvalidParentheses(")()(e()))))))((((")
print(res)


""" AC dfs """

""" bfs solution """





# 212 
# TLE 
class Solution:
    def findWords(self, board, words):
        res = []
        for w in words:
            if self.exists(board, w):
                res.append(w)
        return res

    def exists(self, b, w):
        if not b:
            return False
        n, m = len(b), len(b[0])
        for i in range(n):
            for j in range(m):
                if self.find_path(i, j, b, w, 0):
                    return True
        return False

    def find_path(self, i, j, b, w, k):
        if i < 0 or i >= len(b) or j < 0 or j >= len(b[0]) or b[i][j] != w[k]:
            return False

        if k == (len(w)-1):
            return True

        tmp = b[i][j]
        b[i][j] = '#'
        found = (self.find_path(i+1, j, b, w, k+1) or 
                self.find_path(i-1, j, b, w, k + 1) or
                self.find_path(i, j+1, b, w, k+1) or
                self.find_path(i, j-1, b, w, k+1))
        """ for this problem, we need to reuse the same board, so for the true case, we also need to recover the # back to original value """
        b[i][j] = tmp
        return found



""" implement trie solution """




# 37 
class Solution:
    def solveSudoku(self, board):
        self.fill(board)

    def fill(self, board):
        for i in range(9):
            for j in range(9):
                if board[i][j] == '.':
                    for k in ['1', '2', '3', '4', '5', '6', '7', '8', '9']:
                        # if k is valid for current cell, then fill the rest, if it works return true else, put '.' back and try other digits
                        if self.is_valid(board, i, j, k):
                            board[i][j] = k
                            if self.fill(board):
                                return True
                            else:
                                board[i][j] = '.'
                    return False
        return True

    def is_valid(self, board, i, j, k):
        for x in range(9):
            """ there are some . in the cells also.
            the condition not false including cur cell is . or cur cell not equal to k 
            so the opposite is board[i][x] != '.' and board[i][x] == k
            """
            #check col
            if board[i][x] != '.' and board[i][x] == k:
                return False
            #check row
            if board[x][j] != '.' and board[x][j] == k:
                return False

            #check box
            """ think about the subbox as flat-out array
            3 * (i // 3) to get the starting row, 3 * (j // 3) to get starting colomn 
            x//3 to increment the row, when x <= 3 row will not increment,
            x%3 to find the colmn 
            """
            if board[3 * (i // 3) + x // 3][3 * (j // 3) + x % 3] != '.' and board[3 * (i // 3) + x // 3][3 * (j // 3) + x % 3] == k:
                return False
        return True

# clean version
class Solution(object):
    def solveSudoku(self, board):
        self.fill(board)

    def fill(self, board):
        for i in range(9):
            for j in range(9):
                if board[i][j] == '.':
                    for d in ['1', '2', '3', '4', '5', '6', '7', '8', '9']:
                        if self.check(i, j, board, d):
                            board[i][j] = d
                            if self.fill(board):
                                return True
                            board[i][j] = '.'
                    return False
        return True

    def check(self, i, j, board, d):
        for x in range(9):
            if board[x][j] == d:
                return False
            if board[i][x] == d:
                return False
            if board[3 * (i // 3) + x // 3][3 * (j // 3) + x % 3] == d:
                return False
        return True


""" bitmask solution optimized """
# https: // leetcode.com/problems/sudoku-solver/discuss/15796/Singapore-prime-minister-Lee-Hsien-Loong's-Sudoku-Solver-code-runs-in-1ms



# 51
class Solution(object):
    def solveNQueens(self, n):
        res = []
        queen_cols = [-1]*n # marks the col i-th queen is placed
        self.dfs(n, 0, queen_cols, [], res)
        return res

    # idx means row and also means idx-th queen
    # using call stack frames to represent rows
    def dfs(self, n, idx, cols, path, res):
        # goal is placing n queens in n x n board
        if idx == n:
            res.append(path)
            return
        # choices are n columns, each queen will take a column
        for i in range(n):
            # means put idx-th queen at column i
            cols[idx] = i 
            if self.is_valid(cols, idx):
                tmp = '.' * n
                self.dfs(n, idx+1, cols, path + [tmp[:i] + 'Q' + tmp[i+1:]], res)

    # check vertical and diagonal
    def is_valid(self, cols, r):
        for i in range(r):
            ''' 
            cols[i] == cols[r] means r-th queen col is the same as i-th queen col
            abs(cols[i] - cols[r]) == r - i means r-th queen and i-th queen col and row distance are the same, thus they are on the same diagnals.
            '''
            if cols[i] == cols[r] or abs(cols[i] - cols[r]) == r - i:
                return False
        return True
