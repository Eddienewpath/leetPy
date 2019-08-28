from collections import deque
class Solution:
    def maxAreaOfIsland(self, grid: List[List[int]]) -> int:
        row, col = len(grid), len(grid[0])
        # store seen tuple coordinates 
        seen = set()
        ans = 0
        for r in range(row): 
            for c in range(col):
                if grid[r][c] and (r, c) not in seen:
                    stack = deque([(r, c)])
                    seen.add((r, c))
                    area = 0
                    while stack: 
                        cur_row, cur_col = stack.popleft() # pathon unpacking 
                        area += 1
                        # four possible directions 
                        for neig_row, neig_col in (cur_row+1, cur_col), (cur_row-1, cur_col), (cur_row, cur_col+1), (cur_row, cur_col-1):
                            if 0 <= neig_col < col and 0 <= neig_row < row and (neig_row, neig_col) not in seen and grid[neig_row][neig_col]:
                                stack.appendleft((neig_row, neig_col))
                                seen.add((neig_row, neig_col))
                    ans = max(ans, area)
        
        return ans

                        





