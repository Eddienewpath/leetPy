def numIslands_union(grid):
    n = len(grid)
    if not grid or not n:return 0
    m = len(grid[0])
    uf = UF(n, m, grid)

    for i in range(n):
        for j in range(m):
            if grid[i][j] == '1':
                p = m*i + j
                # right
                q = m*i + j + 1
                if j + 1 < m and grid[i][j+1] == '1':
                    if not uf.connected(p, q):
                        uf.union(p, q)

                # down
                q = m*(i+1) + j
                if i+1 < n and grid[i+1][j] == '1':
                    if not uf.connected(p, q):
                        uf.union(p, q)
    return uf.get_count()




class UF(object):
    def __init__(self, n, m, grid):
        self.count = 0
        self.size = [1 for _ in range(n*m)]
        self.id = [i for i in range(n*m)]

        for i in range(n):
            for j in range(m):
                if grid[i][j] == '1':
                    self.count += 1

    def find_root(self, p):
        while p != self.id[p]:
            # compression
            self.id[p] = self.id[self.id[p]]
            p = self.id[p]
        return p

    def connected(self, p, q):
        return self.find_root(p) == self.find_root(q)

    def get_count(self):
        return self.count

    # weighted
    def union(self, p, q):
        p_root = self.find_root(p)
        q_root = self.find_root(q)

        if self.size[p_root] > self.size[q_root]:
            self.id[q_root] = p_root
            self.size[p_root] += self.size[q_root]
            self.count -= 1
        else:
            self.id[p_root] = q_root
            self.size[q_root] += self.size[p_root]
            self.count -= 1
