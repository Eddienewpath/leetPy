# 841 
""" dfs connectted components """
def canVisitAllRooms(self, rooms):
    room_cnt = len(rooms)
    visited = [0] * room_cnt
    self.dfs(rooms, 0, visited) # starts from any room, if they connected, result will be right
    return sum(visited) == room_cnt


def dfs(self, rooms, i, visited):
    if i < 0 or i >= len(rooms) or visited[i]: return 
    
    visited[i] = True
    for j in rooms[i]: 
        self.dfs(rooms, j, visited)
            


# 1202
""" tip: find all the components chars and their indices, assign the small char to the small index """
# also could be soved using union find
class Solution(object):
    def smallestStringWithSwaps(self, s, pairs):
        n = len(s)
        adj = [[] for _ in range(n)]
        # build the ajacency list/bags
        for i, j in pairs:
            adj[i].append(j)
            adj[j].append(i)

        visited = [False] * n
        s_list = list(s)

        for i in range(n):
            component = []
            self.dfs(i, adj, visited, component)
            component.sort() #sort indices later, the small char will take the index in order
            chars = [s_list[j] for j in component]
            chars.sort()
            for k in range(len(component)):
                s_list[component[k]] = chars[k]
        return ''.join(s_list)


    def dfs(self, i, adj, visited, comp):
        visited[i] = True
        comp.append(i)
        for j in adj[i]:
            if not visited[j]:
                self.dfs(j, adj, visited, comp)


""" union find implementation """




# 1162
""" bfs, find all the 1s and expanding outward simultanously """
class Solution(object):
    def maxDistance(self, grid):
        n = len(grid)
        import collections
        queue = collections.deque()
        # add all the lands to the queue as if they are neighbors. 
        for i in range(n):
            for j in range(n):
                if grid[i][j] == 1:
                    queue.append((i, j))  # all 1s to the queue

        if len(queue) == n*n or len(queue) == 0:
                return -1

        level = 0
        while queue:
            size = len(queue)
            while size:
                i, j = queue.popleft()
                size -= 1
                for x, y in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    x += i
                    y += j
                    # expand the land level by level until not water reached
                    if 0 <= x < n and 0 <= y < n and grid[x][y] == 0:
                        grid[x][y] = 1
                        queue.append((x, y))
            level += 1
        return level-1




# 802
""" graph detecting cycles """
def eventualSafeNodes(self, graph):
    """ for recursion, only worry about current call stack frame logic, coz deeper stack frame uses same logic with differnt input """
    def is_safe(g, i):
        # condition be checked until current node is assigned with a mode 
        if mode[i]:
            # if i's mode is unsafe meaning we encountered a cycle return False, else true 
            return mode[i] == states[0]
        # we initialize current node with unsafe state 
        mode[i] = states[1]
        # check its sucsessors see if they safe or unsafe 
        # if any of its sucessors is unsafe, current node is unsafe
        for j in graph[i]:
            if not is_safe(g, j):
                return False
        # did not find any unsafe node in current node successors, thus current node is safe, assign safe state
        mode[i] = states[0]
        return True

    n = len(graph)
    states = ('safe', 'unsafe')
    mode = [None] * n
    res = []
    for i in range(n):
        if is_safe(graph, i):
            res.append(i)
    return res



# 207 
""" this problem is similar to above problem. using cycle detection in a graph """
def canFinish(self, numCourses, prerequisites):
    course_graph = [[] for _ in range(numCourses)]
    
    for x, y in prerequisites: 
        course_graph[x].append(y)
    
    states = ('can', 'cannot')
    mode = [None] * numCourses
    
    for i in range(numCourses):
        if not self.can_finish(course_graph, mode, i, states):
            return False
    return True 


def can_finish(self, g, mode, i, states):
    if mode[i]: return mode[i] == states[0]
    
    # mark current class as cannot finish 
    mode[i] = states[1]
    # check all its prerequisite courses see if any of them cannot finish return false else mark current course as can finsih and return true 
    for j in g[i]:
        if not self.can_finish(g, mode, j, states): return False
    mode[i] = states[0]
    return True
    
    
        
# 210 
def findOrder(self, numCourses, prerequisites):
        course_graph = [[] for _ in range(numCourses)]
    
        for x, y in prerequisites: 
            course_graph[x].append(y)

        states = ('can', 'cannot')
        mode = [None] * numCourses
        res = []
        for i in range(numCourses):
            if not self.can_finish(course_graph, mode, i, res, states):
                while res: res.pop()
                return res
        return res


def can_finish(self, g, mode, i, res, states):
    if mode[i]: return mode[i] == states[0]

    mode[i] = states[1]
    for j in g[i]:
        if not self.can_finish(g, mode, j, res, states): return False
    
    mode[i] = states[0]
    res.append(i)
    return True

""" implement BFS and bit manipulation solutions  """



# 138 
class Node:
    def __init__(self, x, next=None, random=None):
        self.val = int(x)
        self.next = next
        self.random = random

class Solution(object):
    """ algorithm: 1. cp all the nodes 2. link all the nodes """
    def copyRandomList(self, head):
        if not head: return head
        dic = {}
        n = m = head
        # copy all nodes
        while n:
            dic[n] = Node(n.val)
            n = n.next 
        # linked them up
        while m: 
            """ tip: 
            get() will not throw key error like [] when accessing dictionary
            for this problem, last node will have None as next, thus if dic[None] will throw key error 
            instead get() will return None if None key is given"""
            dic[m].next = dic.get(m.next)
            dic[m].random = dic.get(m.random)
            m = m.next
        return dic[head]


# tree problems below 

# 589 
def preorder(self, root):
        if not root: return []
        res = []
        self.preorder_helper(root, res)
        return res

def preorder_helper(self, r, res):
    res.append(r.val)
    for n in r.children:
        self.preorder_helper(n, res)

# 590 
def postorder(self, root):
    res = []
    if not root: return []
    self.postorder_helper(root, res)
    return res

def postorder_helper(self, r, res):
    for n in r.children:
        self.postorder_helper(n,res)
    res.append(r.val)
    

# 94
def inorderTraversal(self, root):
    stack, res = [], []
    if not root: return res
    while stack or root:
        while root:
            stack.append(root)
            root = root.left
        
        if stack:
            root = stack.pop()
            res.append(root.val)
            root = root.right
    return res 
        

# 144
def preorderTraversal(self, root):
    stack, res = [root],  []
    if not root: return res
    
    while stack:
        root = stack.pop()
        res.append(root.val)
        if root.right: 
            stack.append(root.right)
        if root.left:
            stack.append(root.left)
                
    return res

