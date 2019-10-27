class UnionFind(object):
    def __init__(self, N):
        # each element points to its parent node.  a node is a parent of itself 
        self.id = [i for i in range(N)]
        # size of tree rooted at i. gurrentees lgN find operation
        self.size = [1]*N
   
    # if p and q have same root, they are connected
    def connected(self, p, q):
        return self.find_root(p) == self.find_root(q)

    # weighted union operation: just connect the smaller tree root to the bigger tree root
    def union(self, p, q):
        p_root = self.find_root(p)
        q_root = self.find_root(q)
        # if p tree > q tree size, connect q root to p root and increament size of the bigger tree
        if self.size[p_root] > self.size[q_root]: 
            self.id[q_root] = p_root
            self.size[p_root] += self.size[q_root]
        else: 
            # connect p root to q root
            self.id[p_root] = q_root
            self.size[q_root] += self.size[p_root]

    # path compression: connect every node from p to root directly to the root
    def find_root(self, p):
        # while p's root is not itself
        parent = self.id[p]
        while p != parent:
            # connect p to its parent's parent
            parent = self.id[parent]
            p = parent
        return p





