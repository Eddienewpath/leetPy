# 282
class Solution(object):
    def addOperators(self, num, target):
        """
        :type num: str
        :type target: int
        :rtype: List[str]

         goal: make path result to target
         choices: s[:i] where i in [1, n+1]
  
         prev: last multiply value
         val: current val
         path: res string
        """
        res = []
        self.build_path(num, '', 0, None, res, target)
        return res

    def build_path(self, s, path, val, prev, res, trgt):
        if not s and val == trgt:
            res.append(path)
            return

        for i in range(1, len(s)+1):
            tmp = int(s[:i])
            # prevent starting '01'
            if i == 1 or (i > 1 and s[0] != '0'):
                # cannot write 'if not prev' because prev == 0 will also make this condition work, this way will fail case '105'
                # it will return 1*05, the val is updated correctly, but coz the above condition, will execute if block instead it should 
                # execute else block, thus causing miss operator. 
                if prev is None: 
                    # add first number into path
                    self.build_path(s[i:], path + s[:i], val + tmp, tmp, res, trgt)
                else:
                    self.build_path(s[i:], path + '+' + s[:i], val + tmp, tmp, res, trgt)
                    # need to update the prev with sign, -tmp 
                    self.build_path(s[i:], path + '-' + s[:i], val - tmp, -tmp, res, trgt)
                    # update the prev, and subtract prev from val, then calculate new val
                    self.build_path(s[i:], path + '*' + s[:i], val - prev + prev*tmp, prev*tmp, res, trgt)




# 934
""" # idea hear is that find the fist connected component using dfs and then using bfs to expand this component, once meet the other 
# component return total steps took to get there.  """
class Solution:
    def __init__(self):
        self.directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        import collections
        self.queue = collections.deque()

    def shortestBridge(self, A):
        n, m = len(A), len(A[0])
        visited = [[False for _ in range(m)] for _ in range(n)]
        self._fill(A, visited)
        steps = 0
        """ bfs """
        while self.queue:
            # level size
            size = len(self.queue)
            while size:
                i, j = self.queue.popleft()
                size -= 1
                for x, y in self.directions:
                    # be careful here dont write it like i += x or j += y, this way will completely changed the i an j's value
                    # coz we need to to visit all neibhours of node[i][j]
                    x += i
                    y += j
                    if 0 <= x < len(A) and 0 <= y < len(A[0]) and not visited[x][y]:
                        if A[x][y] == 0:
                            visited[x][y] = True
                            self.queue.append((x, y))
                        else:
                            return steps
            # increment by 1 when all nodes in the same level is traversed. 
            steps += 1
        return -1


    def _fill(self, b, visited):
        n, m = len(b), len(b[0])
        for i in range(n):
            for j in range(m):
                if b[i][j] == 1:
                    # try to find the first component and then return 
                    self._dfs(i, j, b, visited)
                    return

    def _dfs(self, i, j, b, visited):
        if i < 0 or i >= len(b) or j < 0 or j >= len(b[0]) or visited[i][j] or b[i][j] != 1:
            return
        """ this line add all the nodes in the compoent to the queue
        here is the trick, it treats all nodes in the component as one level or as neibhours of each other """
        self.queue.append((i, j))
        visited[i][j] = True
        for x, y in self.directions:
            self._dfs(i+x, j+y, b, visited)




# 752 
def openLock(self, deadends, target):
        deads, visited = set(deadends), set()
        visited.add('0000')
        import collections
        queue = collections.deque()
        queue.append('0000')
        level = 0
        while queue:
            size = len(queue)
            while size:
                """ process block: proces root node here """
                front = queue.popleft()
                if front in deads:
                    size -= 1
                    continue

                if front == target:
                    return level
                """ end of process block """
                # one move on one slot include up and down. there are 4 slots either roll up or roll down
                """ adding neigbors block """
                for i in range(4):
                    roll_down = front[:i] + ('0' if front[i] == '9' else str(int(front[i])+1)) + front[i+1:]
                    roll_up = front[:i] + ('9' if front[i] == '0' else str(int(front[i])-1)) + front[i+1:]

                    if roll_down not in visited:
                        visited.add(roll_down)
                        queue.append(roll_down)

                    if roll_up not in visited:
                        visited.add(roll_up)
                        queue.append(roll_up)
                size -= 1
                """ end of adding neigbors block """
            level += 1
        return -1


# 133
class Node:
    def __init__(self, val=0, neighbors=[]):
        self.val = val
        self.neighbors = neighbors

class Solution:
    """ dfs """

    def cloneGraph(self, node):
        if not node:
            return
        node_cp = Node(node.val)
        cloned = {node: node_cp}
        self.dfs(node, cloned)
        return node_cp

    def dfs(self, node, cloned):
        for nei in node.neighbors:
            if nei not in cloned:
                nei_cp = Node(nei.val)
                cloned[nei] = nei_cp
                cloned[node].neighbors.append(nei_cp)
                self.dfs(nei, cloned)
            else:
                cloned[node].neighbors.append(cloned[nei])


class Solution(object):
    """ bfs """
    def cloneGraph(self, node):
        if not node:
            return
        node_cp = Node(node.val)
        cloned = {node: node_cp}
        self.bfs(node, cloned)
        return node_cp

    def bfs(self, node, cloned):
        import collections
        queue = collections.deque()
        queue.append(node)
        while queue:
            n = queue.popleft()
            for nei in n.neighbors:
                if nei not in cloned:
                    nei_cp = Node(nei.val)
                    cloned[nei] = nei_cp
                    cloned[n].neighbors.append(nei_cp)
                    queue.append(nei)
                else:
                    cloned[n].neighbors.append(cloned[nei])




# 547 
""" connected component in undirected graph. use dfs to find all related friends of current student
tip: for N*N matrix, there are N students. so we need to find their relations. 
because it's a undirected graph, we need to avoid infinit loop by mark the visited student. for example student 
A is frined of B, B is friend of A. if dont mark A as visited, A will be processed again from B. 
"""
class Solution(object):
    def findCircleNum(self, M):
        student_count = len(M)
        cycle_count = 0
        visited = [False] * student_count

        for student in range(student_count):
            if not visited[student]:
                visited[student] = True
                self.find_friends(M, student, visited)
                cycle_count += 1
        return cycle_count

    def find_friends(self, M, student, visited):
        for another_student in range(len(M)):
            if (M[student][another_student] == 1) and (not visited[another_student]):
                visited[another_student] = True
                self.find_friends(M, another_student, visited)




# 695 
def maxAreaOfIsland(self, grid):
    n, m = len(grid), len(grid[0])
    max_area = 0
    for i in range(n):
        for j in range(m):
            max_area = max(max_area, self.dfs(grid, i, j))
    return max_area


def dfs(self, g, i, j):
    # 0 will have 0 area return
    if i < 0 or i >= len(g) or j < 0 or j >= len(g[0]) or g[i][j] != 1: 
        return 0

    g[i][j], cnt = 0, 0 

    for x, y in [(0,1), (0,-1), (1,0), (-1, 0)]:
        cnt += self.dfs(g, i+x, j+y)
    return cnt + 1  




# 733
def floodFill(self, image, sr, sc, newColor):
        if image[sr][sc] == newColor: return image
        self.dfs(image, sr, sc, image[sr][sc], newColor)
        return image


def dfs(self, img, i, j, old, new): 
    if i < 0 or i >= len(img) or j < 0 or j >= len(img[0]) or img[i][j] != old:
        return 
    
    img[i][j] = new
    for x, y in [(0,1),(0,-1), (1, 0), (-1, 0)]:
        self.dfs(img, i+x, j+y, old, new)


""" implement bfs solution """
