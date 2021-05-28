""" continuing graph """

""" tip: count number of components using the characteristc that for all components, its root's parent is itself """
class Solution(object):
    def numIslands(self, grid):
        if not grid:
            return 0
        n, m = len(grid), len(grid[0])
        uf = Union_Find(grid)

        for i in range(n):
            for j in range(m):
                if grid[i][j] == '1':
                    for x, y in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                        x += i
                        y += j
                        if 0 <= x < n and 0 <= y < m:
                            n1, n2 = (i * m + j), (x * m + y)
                            if not uf.connected(n1, n2) and grid[x][y] == '1':
                                uf.union(n1, n2)
        return uf.count



class Union_Find(object):
    def __init__(self, grid):
        n, m = len(grid), len(grid[0])
        self.id = [0] * n * m
        self.size = [1] * n * m
        self.count = 0
        for i in range(n):
            for j in range(m):
                if grid[i][j] == '1':
                    id = i * m + j
                    self.id[id] = id
                    """ only count components with 1 in the grid """
                    self.count += 1

    def connected(self, p, q):
        return self.find(p) == self.find(q)


    def union(self, p, q):
        p_root = self.find(p)
        q_root = self.find(q)

        if self.size[p_root] > self.size[q_root]:
            self.id[q_root] = p_root
            self.size[p_root] += self.size[q_root]
        else:
            self.id[q_root] = p_root
            self.size[q_root] += self.size[p_root]
        # reduce component with 1s by 1
        self.count -= 1

        
    """ find root """
    def find(self, p):
        while self.id[p] != p:
            self.id[p] = self.id[self.id[p]]
            p = self.id[p]
        return p

# 399  2
class Solution(object):
    def calcEquation(self, equations, values, queries):
        roots = {}  # the root of each char
        dist = {}  # a -> root, multiplication of factors along the way to root
        res = []

        # initialize the components
        for i in range(len(equations)):
            x, y = equations[i]
            factor = values[i]
            if x not in roots:
                roots[x] = x
                dist[x] = 1.0
            if y not in roots:
                roots[y] = y
                dist[y] = 1.0
            self.union(roots, dist, x, y, factor)

        for p, q in queries:
            """ tip: do not write this conditoon as (p or q) not in roots, this is differnt. the expression inside the parenthesis will evaluated first, than the rest.  """
            if p not in roots or q not in roots:
                res.append(-1.0)
            else:
                if self.find(roots, dist, p) != self.find(roots, dist, q):
                    res.append(-1.0)
                else:
                    # distance from p to root divides distance from q to root will get distance from p to q, which is p / q 
                    res.append(dist[p]/dist[q])
        return res

    # path compression: make the found root as the parent of current node
    # the idea is to flattern the component, so next time revisting the same node, it does not have to traverse intermediate nodes again
    def find(self, r, d, p):
        if r[p] == p:
            return p
        tmp = r[p]
        r[p] = self.find(r, d, r[p])
        d[p] *= d[tmp] # accumulate the distance on the current node, this is why we use recurive version of compression, top-down
        return r[p]

    def union(self, r, d, p, q, f):
        p_r = self.find(r, d, p)
        q_r = self.find(r, d, q)
        if p_r != q_r:
            r[p_r] = q_r
            """ p_r to q_r distance is calculated by following formula, a little bit math   """
            d[p_r] = f * (d[q]/d[p])  



""" bfs solution """

""" dfs solution """