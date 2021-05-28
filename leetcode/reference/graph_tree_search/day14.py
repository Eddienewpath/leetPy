# 332
# greedy + dfs. all tickets at least form a valid path.
# idea: we greedily pick the smaller lexi destination for every departure, eventually 
# we will form smallest lexical route
def findItinerary(self, tickets):
    res = []
    import collections
    graph = collections.defaultdict(list)
    # sorted(tickets)[::-1] will ensure that when we pop item off, will in lexical order.
    for d, a in sorted(tickets)[::-1]:
        graph[d].append(a)

    self.dfs(graph, 'JFK', res)
    return res[::-1]


def dfs(self, graph, src, res):
    while graph[src]:
        self.dfs(graph, graph[src].pop(), res)
    res.append(src)

# another implementation using deque


""" an Eulerian trail (or Eulerian path) is a trail in a finite graph that visits every edge exactly once (allowing for revisiting vertices). """
def findItinerary(self, tickets):
    """
    :type tickets: List[List[str]]
    :rtype: List[str]
    """

    tickets.sort()
    ticket = {}
    for depart, arrival in tickets:
        if depart not in ticket:
            tmp = collections.deque()
            ticket[depart] = tmp
        ticket[depart].append(arrival)

    res = []
    self.build('JFK', ticket, res)
    return res[::-1]

# post-order traversal
def build(self, src, ticket, res): 
    # defualtdict will not throw exception when the src not in ticket, will return None instead
    while src in ticket and ticket[src]:
        cur = ticket[src].popleft()
        self.build(cur, ticket, res)
    res.append(src)
    

# 721 
# union-find
""" first time that using dictionary and string as id to implement union find """ 
class Solution(object):
    def accountsMerge(self, accounts):
        """ use the unique element to represent the parent or id, in this case is email """
        parents = {}
        email_to_name = {}
        # set up email and its root
        for account in accounts:
            name = account[0]
            # for each email in the list, if the email not in parent, assign itself as its parent
            # and assign the name to the email in the email_to_name 
            # do union operation, which is union first to the second argument
            for em in account[1:]:
                if em not in parents:
                    parents[em] = em
                email_to_name[em] = name
                self.union(em, account[1], parents)

        import collections
        # collecting all the emails with same root/parent into one list
        components = collections.defaultdict(list)
        for em in parents.keys():
            r = self.find(em, parents)
            components[r].append(em)

        return [[email_to_name[r]] + sorted(l) for r, l in components.items()]

    def find(self, email, parents):
        while email != parents[email]:
            parents[email] = parents[parents[email]]
            email = parents[email]
        return email

    def union(self, e1, e2, parents):
        parents[self.find(e1, parents)] = parents[self.find(e2, parents)]


""" implement it using dfs """
def accountsMerge(self, accounts):
    pass 




# 737 
# union find using dictionary implementation
""" tip: 
when a function is returning boolean, think about what case makes the situation false, this is usually easier to find the edge cases. """
def areSentencesSimilarTwo(self, words1, words2, pairs):
    if len(words1) != len(words2): return False
    
    sim = {}
    for w1, w2 in pairs:
        if w1 not in sim: 
            sim[w1] = w1    
        
        if w2 not in sim:
            sim[w2] = w2
        
        self.union(w1, w2, sim)
    
    for i in range(len(words1)):
        # cannot use sim to check parents of two words because, the words maybe not in the sim.
        if self.find(words1[i], sim) != self.find(words2[i], sim): 
            return False 
    return True 
    
    
def find(self, w, sim):
    # this line cover the edge cases that, when two words not in the sim or one of words not in the sim. 
    if w not in sim: return w 
    while w != sim[w]:
        sim[w] = sim[sim[w]]
        w = sim[w]
    return w


def union(self, w1, w2, sim):
    sim[self.find(w2, sim)] = self.find(w1, sim)



# 743 
""" 
dijkstra algorithm, is a greedy algorithm, which picked the least weights from all the ajacent
edges. 
priority queue is used to implement the greedy solution, which is to pick the least weight of all adjacent edges
because of the pq will always maintain the min weight edge destination node on the root

this algorithm essentially is bfs with pq instead of normal queue. 

python: pq is initialized by [], and call it using heapq's heappop and heappush, these methods will turn the list into heap first and do the operation later

this algorithm is searching thru all the nodes in the graph and find the shortest path to get to each node

  """
def networkDelayTime(self, times, N, K):
    import heapq, collections
    # initilize the pq using [], [0] is the priority, in this case, is the time
    """ 
    tips: 
        we are using the time to decide the priority, thus the pq tuple the time should be in front of the node
        which is (0, K)
        dykistra shortest path is just bfs using priority queue, don't forget to maintain a visited dictionary to store
        visited nodes and its weight 
        heapq.heappop() will automatically turn list into a min heap.
        don't forget to inclue the queue in the function of heapq.heappush(pq, node)
    """
    pq, s, adj = [(0, K)], {}, collections.defaultdict(list)
    # initialize the node and its adjacent edges
    for u, v, w in times:
        adj[u].append((v, w))

    while pq:
        time, nxt = heapq.heappop(pq)
        # if one node has multiple relaxation values in the heap, only the smallest one geting pop off first,
        # and the node is added to the final result s dicitonary. and the rest will not be in side this if condition
        if nxt not in s:
            # s is storing each visited nodes and its final shortest path is determined
            s[nxt] = time
            for v, w in adj[nxt]:
                # relax all the edges leaving nxt 
                heapq.heappush(pq, (time+w, v))

    # above procedure will store min time to get to each node from src in the graph. 
    # thus the max value will be the shortest time needed to get to certain node in the graph
    # if that node is reached, then other nodes should have no problem being reached within that time. 
    # therefore we return the max value
    return max(s.values()) if len(s.values()) == N else -1


# 787 
# problem is asking for shortest path to one node and no cycle in the graph,  thus no need to store all the nodes in s like above
def findCheapestPrice(self, n, flights, src, dst, K):
        import heapq, collections
        pq, adj = [(0, src, 0)], collections.defaultdict(list) 

        for u, v, w in flights:
            adj[u].append((v, w))

        while pq:
            price, node, hops = heapq.heappop(pq)
            if node == dst:
                return price
            """ logic here is if there is still stops left, we can go to next hop and push all qualified node """
#           Within K Stops, dont have to be k stops exactly
            if hops <= K:
                # push all neigbhors that within stops to the pq.
                for v, w in adj[node]:
                    heapq.heappush(pq, (price+w, v, hops+1))
        return -1



# 59 
def regionsBySlashes(self, grid):
#       scale the grid 3 times bigger     
    n = len(grid)
    m = len(grid[0])
    bigger_grid = [['1' for _ in range(3*n)] for _ in range(3*m)]
    for i in range(n):
        for j in range(m):
            if grid[i][j] == '/':
                bigger_grid[i*3][j*3+2] = '0'
                bigger_grid[i*3+1][j*3+1] = '0'
                bigger_grid[i*3+2][j*3] = '0'
            elif grid[i][j] == '\\':
                bigger_grid[i*3][j*3] = '0'
                bigger_grid[i*3+1][j*3+1] = '0'
                bigger_grid[i*3+2][j*3+2] = '0'    
    
    count = 0 
    for i in range(3*n):
        for j in range(3*m):
            if bigger_grid[i][j] == '1':
                self.dfs(bigger_grid, i, j)
                count += 1
    return count
                

def dfs(self, grid, i, j):
    if i < 0 or i >= len(grid) or j < 0 or j >= len(grid[0]) or grid[i][j] != '1': return
    
    grid[i][j] = '#'
    for x, y in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
        self.dfs(grid, x+i, y+j)
    
    
