import collections
def numIslands_dfs_recur(grid):
    def dfs(i, j, grid):
        if i < 0 or i >= len(grid) or j < 0 or j >= len(grid[0]) or grid[i][j] == '0':
            return
        grid[i][j] = '0'
        # up
        dfs(i-1, j, grid)
        # right
        dfs(i, j+1, grid)
        # down
        dfs(i+1, j, grid)
        # left
        dfs(i, j-1, grid)
    
    if not grid: return 0
    # row
    n = len(grid)
    # column
    m = len(grid[0])
    
    count = 0
    for i in range(n):
        for j in range(m):
            if grid[i][j] == '1': 
                dfs(i, j, grid)
                count += 1
    return count


def numIslands_bfs(grid):
    if not grid: return 0 
    def bfs(i, j, grid):
        queue = collections.deque()
        queue.append([i, j])
        while queue:
            size = len(queue)
            while size:
                front = queue.popleft()
                if 0 <= front[0] < len(grid) and 0 <= front[1] < len(grid[0]) and grid[front[0]][front[1]] == '1':
                    grid[front[0]][front[1]] = '0'
                    # up
                    queue.append([front[0]-1, front[1]])
                    # right
                    queue.append([front[0], front[1]+1])
                    # down
                    queue.append([front[0]+1, front[1]])
                    # left
                    queue.append([front[0], front[1]-1])
                size -= 1
    count = 0
    for i in range(len(grid)):
        for j in range(len(grid[i])):
            if grid[i][j] == '1':
                bfs(i, j, grid)
                count += 1
    return count
               

def numIslands_dfs_iter(grid): 
    if not grid: return 0
    def dfs_iter(i, j, grid):
        stack = collections.deque() 
        stack.append([i, j])
        grid[i][j] = '0'
        while stack: 
            t = stack.pop()
            top = t[:]
            # up
            top[0] -= 1
            if 0 <= top[0] < len(grid) and grid[top[0]][top[1]] == '1':
                stack.append([top[0], top[1]])
                grid[top[0]][top[1]] = '0'
            
            top = t[:]
            # right
            top[1] += 1
            if 0 <= top[1] < len(grid[0]) and grid[top[0]][top[1]] == '1':
                stack.append([top[0], top[1]])
                grid[top[0]][top[1]] = '0'
            
            top = t[:]                       
            # down
            top[0] += 1
            if 0 <= top[0] < len(grid) and grid[top[0]][top[1]] == '1':
                stack.append([top[0], top[1]])
                grid[top[0]][top[1]] = '0'
            
            top = t[:]
            # left
            top[1] -= 1
            if 0 <= top[1] < len(grid[0]) and grid[top[0]][top[1]] == '1':
                stack.append([top[0], top[1]])
                grid[top[0]][top[1]] = '0'  

    count = 0
    for i in range(len(grid)):
        for j in range(len(grid[i])):
            if grid[i][j] == '1':
                dfs_iter(i, j, grid)
                count += 1
    return count






