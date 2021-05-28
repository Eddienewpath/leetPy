# 997 
# judge with n-1 in degrees and 0 out degree
# naive solution
def findJudge_naive(self, N, trust):
    if N == 1: return N
    in_degree = {}
    out_degree = {}
    for a, b in trust:
        in_degree[b] = in_degree.get(b, 0) + 1
        out_degree[a] = out_degree.get(a, 0) + 1

    for k, v in in_degree.items():
        if v == N-1:
            if k not in out_degree:
                return k
    return -1

""" clever solution """
# trick here is to think of the indegree and outdegree as + and -, and find the differences of each node
def findJudge(self, N, trust):
    count = [0] * (N+1)

    for a, b in trust:
        count[a] -= 1
        count[b] += 1

    for i in range(1, N+1):
        if count[i] == N-1:
            return i
    return -1



# 863 
# my implementation. see more concise version below
# idea is the same: first convert the tree into a undirected graph and do a bfs search
import collections
def distanceK(self, root, target, K):
    res = []
    graph = collections.defaultdict(list)
    self.build_graph(root, graph)
    if len(graph.keys()) < K + 1: return res
    self.bfs(graph, target, res, K)
    return res
    
    
def build_graph_old(self, root, graph):
    if not root: return 
    if root.left: 
        graph[root].append(root.left)
        graph[root.left].append(root)
    if root.right:
        graph[root].append(root.right)
        graph[root.right].append(root)

    self.build_graph(root.left, graph)
    self.build_graph(root.right, graph)
    

def bfs(self, graph, src, res, K):
    queue = collections.deque()
    queue.append(src)
    level = 0
    visited = [False] * len(graph.keys())
    while queue:
        
        if level == K:
            while queue:
                res.append(queue.popleft().val) 
                
        size = len(queue)
        while size:
            front = queue.popleft()
            visited[front.val] = True
            for n in graph[front]:
                if not visited[n.val]:
                    queue.append(n)
            size -=1 
        level += 1

#another version of my implementaion
def distanceK(self, root, target, K):
    adj, res = collections.defaultdict(list), []
    self.build_graph(None, root, adj)
    queue = collections.deque()
    queue.append(target.val)
    visited = set()
    dist = 0
    while queue:
        size = len(queue)
        while size:
            front = queue.popleft()
            visited.add(front)
            if dist == K:
                res.append(front)
            for nei in adj[front]:
                if nei not in visited: 
                    queue.append(nei)
            size -= 1
        dist += 1
    return res
                    

""" lee's implementation conscise but a little slow """
def distanceK(self, root, target, K):
    graph = collections.defaultdict(list)
    self.build_graph(root, graph)
    bfs = [target.val]
    seen = set(bfs)
    for i in range(K):
        # for every node in bfs add its unseen neibhours to the new bfs list
        bfs = [y for x in bfs for y in graph[x] if y not in seen]
        seen |= set(bfs) #using set theory 'or' to combine two sets of elements
    return bfs


def build_graph_old(self, root, graph):
    if not root: return 
    if root.left: 
        graph[root.val].append(root.left.val)
        graph[root.left.val].append(root.val)
    if root.right:
        graph[root.val].append(root.right.val)
        graph[root.right.val].append(root.val)

    self.build_graph(root.left, graph)
    self.build_graph(root.right, graph)
    
""" cleaner version of building a graph """


def build_graph(self, parent, r, adj):
        if not r:
            return
        if r.left:
            adj[r.val].append(r.left.val)
        if r.right:
            adj[r.val].append(r.right.val)
        if parent:
            adj[r.val].append(parent.val)
        self.build_graph(r, r.left, adj)
        self.build_graph(r, r.right, adj)

# 785 
""" 
bipartite: for given graph, if all the nodes in the graph can be divided into two subsets, A and B, 
and for every edge in the graph, one of its node is in A, another is in B.

trick: using coloring technique, if bipartite graph, there will be no contradiction. 
"""
def isBipartite(self, graph):
    color = {}
    # graph maybe disconnected, meaning there are multiple sub-graphs might be bipartite
    for i in range(len(graph)):
        # must has this conditon, because color can be set during previous dfs
        if i not in color:
            color[i] = 0
            if not self.check_color(graph, i, color):
                return False
    return True


def check_color(self, graph, src, color):
    for n in graph[src]:
        """ if neigbor is same color as the src node, then return False  else, set the neigbor node to opposite color
        and repeat the process for neibhor and its neighbor"""
        if n in color:
            if color[n] == color[src]: 
                return False
        else:
            # trick to assign opposite color
            color[n] = 1 - color[src]
            if not self.check_color(graph, n, color):
                return False
    return True
            
    
    
