class ListNode:
    def __init__(self, k, v):
        self.key_val_pair = (k, v)
        self.next = None 
    
""" for simplicity, assume our key and value pair is int and int """  
class HashTable:
    def __init__(self):
        self.capacity = 256
        self.buckets = [None]*self.capacity
    

    def put(self, k, v):
        idx = k % self.capacity
        cur = self.buckets[idx]
        if not cur:
            # be careful, dont assign to cur, which will not sync to buckets[idx]
            self.buckets[idx] = ListNode(k, v)
        else:
            while True:
                if cur.key_val_pair[0] == k:
                    # remember tuple is not mutable, so we need to assign a new tuple pair to the existing key
                    cur.key_val_pair = (k, v)
                    return
                if not cur.next: break
                cur = cur.next 
            cur.next = ListNode(k, v)


    def get(self, k):
        idx = k % self.capacity
        cur = self.buckets[idx]
        while cur: 
            if cur.key_val_pair[0] == k:
                return cur.key_val_pair[1]
            cur = cur.next
        return -1

        
    def remove(self, k):
        idx = k % self.capacity
        cur = self.buckets[idx]

        if not cur: return
        else:
            if cur.key_val_pair[0] == k:
                # move the head to the next element, so the original head will be unreferenced, and get garbage collected. 
                self.buckets[idx] = cur.next
                return
            run = cur.next
            prev = cur 
            while run: 
                if run.key_val_pair[0] == k:
                    prev.next = run.next
                    return
                prev, run = run, run.next
        
# ht = HashTable()
# ht.put(1,1)
# ans = ht.get(1)
# print(ans)


""" basic idea """
""" 
using an array to map the subtree root and parent/root
if two nodes share same root, meaning they are connected
we can connect two unconnected graphs by adding the root of smaller graph directly to the 
bigger graph root, so to make the graph shorter
the find_root method using the path compression to compress the path along the search of root
and making the graph more flat-out 
path compression: place current node to its parent's parent
"""
class UnionFind:
    def __init__(self, N):
        self.id = [i for i in range(N)]
        self.size = [1] * N


    def connected(self, p, q):
        return self.find_root(p) == self.find_root(q)


    def union(self, p, q):
        p_root = self.find_root(p)
        q_root = self.find_root(q)

        if self.size[p_root] < self.size[q_root]:
            self.id[p_root] = q_root
            self.size[q_root] += self.size[p_root]
        else:
            self.id[q_root] = p_root
            self.size[p_root] += self.size[q_root]


    # set current node's parent equal to parent's parent
    def find_root(self, p):
        while p != self.id[p]:
            self.id[p] = self.id[self.id[p]]
            p = self.id[p]
        return p


# uf = UnionFind(5)
# uf.union(0, 1)
# uf.union(1, 2)
# uf.union(1, 3)
# uf.union(1, 4)

# print(uf.find_root(0))
# print(uf.connected(0, 4))

""" 
Max_Heap implementation is for building a heap using given array, mainly for usage of heap sort 

the following priority queue class has more functionalities. you form heap when you insert an item or when you pop item, it will automatically maintain heap property
"""
class Max_Heap:
    def __init__(self, arr):
        self.arr = arr
        self.heap_size = len(arr)
    
    # bottom up 
    def build(self):
        # start from the right most node of the level right above the leaves
        start = len(self.arr)//2 - 1
        for i in range(start, -1, -1):
            self.heapify(i)
    
    # top down
    def heapify(self, i):
        if i >= self.heap_size: 
            return 
        
        left, right, max_idx = 2*i+1, 2*i+2, -1

        if left < self.heap_size and self.arr[left] > self.arr[i]:
            max_idx = left
        else:
            max_idx = i
        
        if right < self.heap_size and self.arr[right] > self.arr[max_idx]:
            max_idx = right
        
        if max_idx != i: 
            self.arr[i], self.arr[max_idx] = self.arr[max_idx], self.arr[i]
            self.heapify(max_idx)

# in-place sorting, O(nlgn), not stable coz when array is large, the comparisons will be all over the memory.
# merge_sort need O(n) space, but stable.
# quicksort on average is the best sorting.
def heap_sort(nums):
    max_heap = Max_Heap(nums)
    max_heap.build()
    for i in range(len(nums)-1, 0, -1):
        nums[0], nums[i] = nums[i], nums[0]
        max_heap.heap_size -= 1
        max_heap.heapify(0)
        

# nums = [7,3,5,2,9,1]
# heap_sort(nums)
# print(nums)


""" 
priority queue can be used in computer operating system job scheduling and choose the job with higher priority to process first
we implement priority queue using the concept of heap. for simplicity we assume the capacity of the heap, for industrial implementation, we can use 
resizing array to avoid overflowing.
"""
class MaxPriorityQueue:
    def __init__(self, capacity):
        self.pq = [None]*capacity # total capacity of pq, that a pq can store. 
        self.N = 0 # number of elements are inside the heap
    
    """
    insert an item into the heap procedures:
    -first append the item to the end of the heap, assume we are within the capacity
    -do swim up operation to find its right position 
    -increment the N
    """
    def insert(self, item):
        if self.N != len(self.pq): 
            if self.N == 0: 
                self.pq[0] = item   
            else:
                self.pq[self.N] = item
                self._swim_up(self.N)
            self.N += 1

    """
    exchange pq[0] and pq[N-1]
    decrement N by 1 
    _sink(pq[0])
    null pq[N-1]
    return the pq[0] element 
    """
    def pop(self):
        if not self.isEmpty():
            max_item = self.pq[0]
            self.exchange(0, self.N-1)
            self.pq[self.N-1] = None
            self.N -= 1
            self._sink(0)
            return max_item

    """
    compare with item's parent, do exchange operation if cur item is greater than parent 
    """
    def _swim_up(self, i):
        while True:
            p = (i+1)//2 - 1
            if p >= 0 and self.pq[p] < self.pq[i]:
                self.exchange(p, i)
                i = p
                continue
            break
    
    
    """
    compare cur item with its children and _sink down to its correct position  
    """            
    def _sink(self, i):
        while True:
            left = 2 * i + 1 
            right = left + 1 

            k = right if right < self.N and self.pq[right] > self.pq[left] else left
            if k < self.N and self.pq[k] > self.pq[i]:
                self.exchange(i, k)
                i = k
                continue
            break



    def isEmpty(self):
        return self.N == 0


    def exchange(self, i, j):
        self.pq[i], self.pq[j] = self.pq[j], self.pq[i]


    def __repr__(self): 
        return str(self.pq)



# pq = MaxPriorityQueue(10)
# pq.insert(1)
# for i in range(2, 6):
#     pq.insert(i)
# print(pq.pop())
# print(pq)
# print(pq.pop())
# print(pq)


"""
undirect graph is actually a direct graph with a pair node pointing to each other
undirect graph
V: number of vertices 
adj: store each vertex's ajacents

implemented using ajacency list

all the graph processing algo is decoupled 
"""
class Undirected_Graph: 
    def __init__(self, V):
        self.V = V 
        self.adj = [[] for _ in range(self.V)]
    
    # v, w are vertices 
    def add_edge(self, v, w): 
        self.adj[v].append(w)
        self.adj[w].append(v) # comment this out will be the digraph class.


    def get_adj(self, v):
        return self.adj[v]

    def size(self):
        return self.V



""" three most common graph processing algorithms  """

class DFS_Paths:
    def __init__(self, g, s):
        self.visited = [False for _ in range(g.size())]
        self.edge_to = [None] * g.size() # previous vertex
        self.s = s 
        self.dfs(g, s)

    def dfs(self, g, v):
        self.visited[v] = True
        for w in g.get_adj(v):
            if not self.visited[w]: 
                self.edge_to[w] = v
                self.dfs(g, w)
        

    def has_path_to(self, v):
        return self.visited[v]


    def path_to(self, v):
        if not self.has_path_to(v): return None
        path = []
        while v != self.s: 
            path.append(v)
            v = self.edge_to[v]
        path.append(self.s)
        return path
        


class BFS_Paths:
    def __init__(self, g, s): 
        self.s = s
        self.visited = [False for _ in range(g.size())]  
        self.edge_to = [None] * g.size()
        self.bfs(g, s)

    
    def bfs(self, g, v):
        from collections import deque
        q = deque()
        q.append(v)
        self.visited[v] = True
        while q: 
            front = q.popleft()
            for w in g.get_adj(front):
                if not self.visited[w]:
                    q.append(w)
                    self.visited[w] = True
                    self.edge_to[w] = front


    def has_path_to(self, v):
        return self.visited[v]


    def path_to(self, v):
        if not self.has_path_to(v):
            return None
        path = []
        while v != self.s:
            path.append(v)
            v = self.edge_to[v]
        path.append(self.s)
        return path



class ConnectedComponents:
    def __init__(self, g):
        self.visited = [False for _ in range(g.size())]
        self.id = [None for _ in range(g.size())]
        self.count = 0
        for w in range(g.size()):
            if not self.visited[w]: 
                self._dfs(g, w)
                self.count += 1 


    # number of conencted components in the graph 
    def get_count(self):
        return self.count

    # each vertex if they are in the same component they share same id
    def get_id(self, v):
        return self.id[v] 

    # check if v and w is connected or not
    def connected(self, v, w):
        return self.id[v] == self.id[w] 


    def _dfs(self, g, v):
        self.visited[v] = True
        self.id[v] = self.count # mark all same components in same id
        for w in g.get_adj(v):
            if not self.visited[w]:
                self._dfs(g, w)


# g = Undirected_Graph(6)
# g.add_edge(2, 3)
# g.add_edge(2, 5)
# g.add_edge(0, 1)
# # g.add_edge(1, 2)
# g.add_edge(3, 4)
# g.add_edge(4, 5)

# print(g.get_adj(2))  # [3, 5]
# print(g.size())

# 0-1 2-3-4
#     |  /
#      5

# d = DFS_Paths(g, 0)
# l = d.path_to(5)

# b = BFS_Paths(g, 0)
# l = b.path_to(5)
# print(l)

# c = ConnectedComponents(g)
# print(c.get_count())
# print(c.connected(0, 5))
# print(c.get_id(0))
# print(c.get_id(5))


""" 
graph challeges 
Is the graph bipartite? 
find a cycle in the graph. 
find a cycle that uses every edge exactly once
find the cycle visits every vertex exactly once, this is impossible to find the efficient algoritm. np complete problem. 
"""


""" shortest path with weight in directed graph """

class Weighted_Directed_Edge:
    def __init__(self, v, w, weight):
        self.v = v 
        self.w = w 
        self.weight = weight

    def frm(self):
        return self.v 


    def to(self):
        return self.w 


    def weight(self):
        return self.weight 


class Edge_Weighed_Digraph:
    def __init__(self, V):
        self.V = V 
        self.adj = [[] for _ in range(self.V)]


    def add_edge(self, e):
        v = e.frm()
        self.adj[v].append(e)

    def get_adj(self, v):
        return self.adj[v] 

    def size(self):
        return self.V



class Trie:
    pass


