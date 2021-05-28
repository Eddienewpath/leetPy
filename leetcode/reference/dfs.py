""" 
dfs
goal: all green + some yellow
"""
# p200 
def numIslands(grid):
    if not grid: return 0
    # visit four directions and set visited 1's to 0's 
    def dfs(grid, i, j):
        if i < 0 or j < 0 or i >= len(grid) or j >= len(grid[0]) or grid[i][j] != '1': return
        grid[i][j] = '0'
        directions = {(-1, 0), (1,0), (0, -1), (0, 1)}
        for x, y in directions:
            dfs(grid, i+x, j+y)

    n = len(grid)
    m = len(grid[0])
    cnt = 0
    for i in range(n):
        for j in range(m):
            if grid[i][j] == '1':
                dfs(grid, i, j)
                cnt += 1 
    return cnt


# p104
# I always think the terminating case is when a node doesnt have left and right children and return 1, but actually this case can be handled
# by terminating condition if not root: return 0. because we will add 1 at the last line. 
def maxDepth(root):
    if not root: return 0
    left = self.maxDepth(root.left)
    right = self.maxDepth(root.right)
    return max(left, right) + 1


# p108 
# find the root, and build the left and build the right and connect root with left and right
def sortedArrayToBST(nums):
    if not nums: return None
    mid = len(nums)//2
    root = TreeNode(nums[mid])
    left = sortedArrayToBST(nums[ :mid])
    right = sortedArrayToBST(nums[mid+1: ])
    root.left = left 
    root.right = right
    return root



# 301
# using two stacks one to store invalid chars and one for result
# if matched parenthesis will be pop off invalid stack and added in to result. 
# question ask for all possible results
def removeInvalidParentheses(s):
    pass
    # if not s: return ''
    # invalid, valid = [], []
    # for c in s:
    #     if not invalid and c == ')':
    #         invalid.append(')')
    #     elif c == ')' and invalid[-1] == '(':
    #         valid.append(invalid.pop())
    #         valid.append(c)
    #     else:
    #         invalid.append(c)
    # return ''.join(valid)


# p100 
def isSameTree(p, q):
    if not p and not q: return True
    if not p or not q: return False
    left = isSameTree(p.left, q.left)
    right = isSameTree(p.right, q.right)
    return p.val == q.val and left and right


# p394 
# key to solve this problem is realize when to push and when to pop and what is being stored in the stack and why do you want to store this 
# for this problem, you need to store the repeating number and the string that is not being decoded before the '['
# thus when we see the '[', we need to push previous string and repeating number onto their own stack.
# when we see ']', we need to pop off previous string and append the current collected string and repeating given time you stored on the stack. 
def decodeString_iter(s):
    if not s: return ''
    n = len(s)
    num_stack , str_stack = [], []
    idx , cur = 0, ''
    while idx < n: 
        if s[idx].isdigit():
            d = 0 
            while s[idx].isdigit():
                d = 10*d + int(s[idx])
                idx += 1
            num_stack.append(d)
        elif s[idx] == '[':
            str_stack.append(cur)
            # reset cur once cur is stored in the stack
            cur = ''
            idx += 1
        elif s[idx] == ']':
            prev = str_stack.pop()
            repeat = num_stack.pop()
            # append decode string to the prev string.
            cur = prev + cur*repeat
            idx += 1 
        else: 
            cur += s[idx]
            idx += 1
    return cur
     
# recursive solution:
def decodeString_recur(s):
    if len(s) == 0: return ''
    idx, n, res = 0, len(s), []
    while idx < n:
        if s[idx].isdigit():
            d = 0
            while s[idx].isdigit():
                d = 10*d + int(s[idx])
                idx += 1
            idx += 1
            start = idx
            cnt = 1
            # match the right closing bracket
            while cnt != 0:
                if s[idx] == '[':
                    cnt += 1
                if s[idx] == ']':
                    cnt -= 1
                idx += 1
            idx -= 1
            end = idx  # exclusive
            decoded = self.decodeString(s[start: end])
            res.append(decoded*d)
            idx += 1
        else:
            res.append(s[idx])
            idx += 1
    return ''.join(res)


# p695 
# dfs: define clearly what does this function do recursively.
# the grid stores integers 1 and 0
def maxAreaOfIsland(grid):
    # dfs return the area that start at [i, j]
    def dfs(grid, i, j):
        if i < 0 or j < 0 or i >= len(grid) or j >= len(grid[0]) or grid[i][j] == 0: return 0 
        grid[i][j] = 0
        directions = [(1,0), (-1, 0), (0, 1), (0, -1)]
        res = 0
        for x, y in directions:
            res += dfs(grid, i+x, j+y)
        return res+1

    n, m = len(grid), len(grid[0])
    area = 0
    for i in range(n):
        for j in range(m):
            if grid[i][j] == 1:
                area = max(dfs(grid, i, j), area)
    return area


# p105 
# find the root, build the left and right subtree and connect root with the left and  right trees
# preorder = [3, 9, 20, 15, 7]
# inorder = [9, 3, 15, 20, 7]
# this problem can be optimized by store index of inorder array to a dictionary for fast retrieval 
def buildTree(preorder, inorder):
    if not preorder or not inorder: return None
    root = TreeNode(preorder[0])
    idx = inorder.index(preorder[0])
    n = len(inorder)
    left_size = len(inorder[:idx])
    right_size = len(inorder[idx+1:])
    left = self.buildTree(preorder[1: left_size+1], inorder[:idx])
    right = self.buildTree(preorder[left_size+1: left_size+1+right_size], inorder[idx+1:])
    root.left = left
    root.right = right
    return root


# 110:
# top down for every node compare their height diff if exceeds 1 than return false
# time complexity O(n^2) because calculate height for every node worse case is O(n)
# worse case for this problem is this tree is not balanced, so for every node you have O(n) to check height
# so the time is O(n^2)
def isBalanced(root):
    def height(root):
        # height is the number of edges from root to leaf 
        if not root: return -1
        return max(height(root.left), height(root.right)) + 1
    if not root: return True
    if abs(height(root.left) - height(root.right)) > 1: return False
    return isBalanced(root.left) and isBalanced(root.right)


# optimal solution: 
# bottom up. using a flag to indicate whether there is a level that causes unbalanced so we can pass up the flag inidcating it
# so we can do check every node only once during height calculcation
# time complextiy is O(n)
def isBalanced_optimal(root):
    # we can calculate height using number of nodes along the way, thus return 0 at base caes
    def height_balance_checker(root):
        if not root: return 0
        left = height_balance_checker(root.left)
        right = height_balance_checker(root.right)
        # pass up the indicator 
        if left == -1 or right == -1 or abs(left - right) > 1: return -1 
        return max(left, right)+1
    
    if not root: return True 
    return height_balance_checker(root) != -1 


# push all left child to the stack until left most, then we found the left most child on the tree
# we can start calculate depth of each node and store in the dictionary. if right subtree is null or was processed before
# we can process current root depth and store it. 
def isBalanced_iter(root):
    stack, r, depth, last = [], root, {}, None
    while r or stack:
        if r:
            stack.append(r)
            r = r.left
        else:
            r = stack[-1]
            # if right tree is null or right tree  is already calculated or processed (last == r.right), then we can store current r's depth in the depth dict
            # we did pop the last node off the stack but current root still has reference to the last node, so the else block can still go there thus cause initinitet loop  
            if not r.right or last == r.right:
                r = stack.pop()
                left = depth.get(r.left, 0)
                right = depth.get(r.right, 0)
                if abs(left - right) > 1: return False
                depth[r] = max(left, right) + 1
                # update last node processed, prevent repeating
                last = r
                # prevent code goes into if block repeating process
                r = None 
            else:
                r = r.right
    return True


# p101 
# base case: if two children are none, thus return true, if one of the children is none, return false
# for dfs problem think about the right ending or terminating case. 
# this problem is hard to think of using a helper function change the param lists. 
def isSymmetric(root):
    if not root: return True
    def helper(r1, r2):
        if not r1 and not r2:
            return True
        if not r1 or not r2:
            return False
        return r1.val == r2.val and helper(r1.left, r2.right) and helper(r1.right, r2.left)
    return helper(root.left, root.right)

# left and right in this problem is representing (previous root1's left and root2's right) and (previous root1's right and root2's left)
# so if two nodes are both none, if can still form a symetric tree, thus we continue. if one of them are none, is definitly false. 
def isSymmetric_iter(root):
    if not root: return True
    stack = [root.left, root.right]
    while stack:
        left, right = stack.pop(), stack.pop()
        if not left and not right: continue
        if not left or not right: return False
        if left.val != right.val: return False
        if left.val == right.val:
            stack.append(left.right)
            stack.append(right.left)
            stack.append(left.left)
            stack.append(right.right)
    return True



# p199 
# recursive idea: find all right side view list for left and right subtrees and combine their results.
# solution aspired by stephan pochman
# O(n^2)
def rightSideView_bf(root):
    # bases case: if null tree, there is nothing on the right side so return empty list
    if not root: return []
    right = rightSideView_bf(root.right)
    left = rightSideView_bf(root.left)
    return [root.val] + right + [left[len(right): ]]


# O(n)
import collections
def rightSideView_bfs(root):
    if not root: return []
    queue = collections.deque()
    queue.append(root)
    res = []
    while queue:
        res.append(queue[-1].val)
        size = len(queue)
        while size:
            front = queue.popleft()
            if front.left: queue.append(front.left)
            if front.right: queue.append(front.right)
            size -= 1 
    return res 


# dfs from right to left and collect the one when level equal to size of result
# intuition: find the right most node that will increase the level
def rightSideView_dfs_stephan(root):
    def collect(root, level, right_view):
        if root:
            if level == len(right_view):
                right_view.append(root.val)
            # order of recursive call is matter
            collect(root.right, level+1, right_view)
            collect(root.left, level+1, right_view)
    res = []
    collect(root, 0, res)
    return res 


def rightSideView_dfs(root):
    def helper(root, level, res):
        if root: 
            # reach a new level, add the first node to the new list
            if level == len(res):
                tmp = [root.val]
                res.append(tmp)
            # if current node is not at the deeper level, meaning it is on the same level, so we can add it to the current level list
            if level < len(res):
                res[level].append(root.val)
            level += 1 
            helper(root.left, level, res)
            helper(root.right, level, res)

    res = []
    if not root: return []
    helper(root, 0, res)
    return [a[-1] for a in res]


# p114
# how to update the last node of flatten left tree. 
# to avoid traversing the flatten left tree to the last node, so that we can connect last node with flatten right tree
# the solution is to do a reversed preorder traversal(right -> left -> root) and maintain a global reference to indicate the next node to connect. 
# we can update the next node when visiting the node, thus O(n)
# recursive problem tips: thinking about null tree -> one node tree -> two node tree -> three node tree -> entire tree
 
next = None 
def flatten_recur(root):
    if root: 
        # flatten right tree first then left tree, s.t. we can avoid traversing a list of nodes to get the last node. 
        flatten(root.right)
        flatten(root.left)
        # connect the left and right tree together 
        root.right = next
        root.left = None 
        # update the next node to the current root node, climbing up the linked list reversely when processing a node. 
        next = root


# using iterative preorder traversal to build the list. 
# there are two ways to iterate a b-tree, push all left side all in once on to the stack or only push two children of the root at once
# because we need to build the list use the right link, so we need to store the right child first so that the right child 
# will not be replaced. so we shall choose second version of preorder traversal. and build the list along the way. 
# this is not in place traversal
def flatten_iter(root):
    if root:
        stack = [root]
        while stack:
            r = stack.pop()
            if r.right: stack.append(r.right)
            if r.left: stack.append(r.left)
            # if stack is not empty.
            if stack: r.right = stack[-1]
            r.left = None
    
# there is a better implementation called morris in place traversal. 
def flatten_iter(root):
    pass 


# 959
# intuition: when you see the given example, it is hard to recognize the area, we need to zoom in 3 times for better view.
# upscale the grid. usig 3*3 grid to represent '\' and '/' and problem become number of islands problem
# create the new grid. for the given grid, each string is a row, and length of the string is number of columns. 
def regionsBySlashes(grid):
    def dfs(grid, i, j):
        if i < 0 or j < 0 or i >= len(grid) or j >= len(grid) or grid[i][j] != '1': return 
        grid[i][j] = '#'
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        for x, y in directions:
            dfs(grid, i+x, j+y)
    
    n = len(grid)
    m = len(grid[0])
    bigger_grid = [['1' for _ in range(3*m)] for _ in range(3*n)]
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
                dfs(bigger_grid, i, j)
                count += 1
    for r in bigger_grid:
        print(r)
    return count 

# 2
# print(regionsBySlashes([" /", "/ "])) 


# 109 
# preorder
# build tree: create root, conenct root with left and right subtree. 
# for balanced bst, we need to find the mid point, which is the root of the tree.
# for linkedlist, find mid point using slow and fast pointers to locate the mid, than update the linkedlist length
def sortedListToBST(head):
    def findMid(head):
        if not head: return None
        prev, slow, fast = head, head, head
        while fast and fast.next:
            prev = slow
            slow = slow.next
            fast = fast.next.next
        prev.next = None
        return slow

    if not head: return 
    if not head.next: return TreeNode(head.val)
    mid = findMid(head)
    root = TreeNode(mid.val)
    left = sortedListToBST(head)
    right = sortedListToBST(mid.next)
    mid.next = None
    root.left = left 
    root.right = right
    return root



# 979
# intuition is that: to make left and right subtree childrens to 1 and give coins to the parent/root and let parent's parent worry about parent's unbalanced amount of coins
res = 0 
def distributeCoins(self, root):
    def dfs(root):
        if not root: return 0
        left = dfs(root.left)
        right = dfs(root.right)
        # total moves needed to make left and right subtree having one coin, for the entire tree, if left and right subtrees all have 1 coin the root will have 1 coin because the total coin is the total of nodes in the tree
        self.res += abs(left) + abs(right) 
        # left and right is the numebr of coin needed or number of extra coins left and right tree have. 
        return root.val - 1 + left + right
    dfs(root)
    return self.res


# 116
# this problem can be easily solved by using bfs using a queue to store all the nodes and connect them level by level
# or we can use stack to do a dfs seach and connect all the nodes together
# idea is that because the tree given is always perfect meaning every node has two children and all leaves are on the same level 
# so, we can connect each level from level to right as long as the current root has left child
# the key to this problem is how to connect two subtrees in the middle 
def connect(root):
    r = root
    while r and r.left:
        cur = r
        # moving horizontally til reach the None
        while cur:
            cur.left.next = cur.right
            # if the there is no subtree on the next, we finished connecting at current level 
            cur.right.next = cur.next.left if cur.next else None
            cur = cur.next
        r = r.left
    return root

# recursive solution with less code
# connect the root and the two subtree roots using next pointer, then recursively connect left subtree and right subtree.
def connect_recur(root):
    def helper(root):
        if root and root.left:
            if root.right:
                root.left.next = root.right 
                root.right.next = root.next.left if root.next else None 
            helper(root.left)
            helper(root.right)
    r = root
    helper(r)
    return root


# p547 
# this is graph subcategory--conencted components problems, mark all nodes in a component as visted and rest if similar to number of island
# important tip is that when student himself is friend circle of himself, this is a indicate that this will be a dfs problem 
# this solution check each student if he is friend with other students or he is in a connected component.
def findCircleNum(self, M):
    def findFriends(m, i, visited):
        for j in range(len(m)):
            if m[i][j] and not visited[j]:
                visited[j] = 1
                findFriends(m, j, visited)
    n = len(M)
    # to marke if current student included in some connected component, if yes, mark him as visited. 
    visited = [0]*n
    count = 0 
    for i in range(n):
        if visited[i] == 0:
            findFriends(M, i, visited)
            count += 1
    return count


# p133 
# dfs
# dic store the current node and its copy 
def cloneGraph(self, node):
    def dfs(node, dic):
        for nei in node.neighbors:
            if nei not in dic:
                nei_cp = Node(nei.val, [])
                dic[nei] = nei_cp
                dic[node].neighbors.append(nei_cp)
                dfs(nei, dic)
            else:
                # if a current nei of node exists alreay, you just need to add into the node_cp list
                dic[node].neighbors.append(dic[nei])
    dic = {node: Node(node.val, [])}
    dfs(node, dic)
    return dic[node]


# p257 
def binaryTreePaths(self, root):
    # dfs definition: building path from root to leaves 
    def dfs(root, res, path):
        # this ending condition could be tricky if you do if not root as ending, will create duplicates paths. 
        if not root.left and not root.right:
            path = path + '->' + str(root.val)
            res.append(path[2:])
            return
        dfs(root.left, res, path + '->' + str(root.val)) if root.left else None
        dfs(root.right, res, path + '->' + str(root.val)) if root.right else None
    res = []
    if not root: return res 
    dfs(root, res, '')
    return res 

# 112
# dfs definition: if there is a path sum up to target sum
def hasPathSum(self, root, sum):
    if not root: return False
    if not root.left and not root.right and sum - root.val == 0:
        return True 
    return hasPathSum(root.left, sum - root.val) or hasPathSum(root.right, sum - root.val)


# 332
# trick: treat all the destinations as children of starting locations, and sort the children and do a postorder traversal, and reverse the result will be the final result
# greedy: sort + post_order, always choose the smaller order destinations. because the problem gurentee using up all the tickets
def findItinerary(self, tickets):
    # post order 
    def visit(src, dic, res):
        while dic[src]:
            # remove the visited node from deque
            front = dic[src].pop()
            visit(front, dic, res)
        res.append(src)
    # key: departure, value: a list of arrivals
    dic = collections.defaultdict(list)
    # python will sort nested list, if two nest list has same first item, it will compare the second one, ....
    # so we sort the tickets from higer lexi order to low lexi order, and then reverse the result from low lexi to hi lexi and append the high lexi arrival 
    # in the list, so that later we can just pop off the list if current node is visted, without using deque. 
    for t in sorted(tickets)[::-1]:
        dic[t[0]].append(t[1])
    res = []
    visit('JFK', dic, res)
    print(["JFK","MUC","LHR","SFO","SJC"])
    return res[::-1]



#p111
# definition: mim nodes from root to leaves
# special case is the bottom of the tree. if a root does not have left or right subtree, then the min depth is the depth of left or right subtree plus one
# if left and right trees exists, then we do min(left, right)+1
# another thing worth mention is that: if we are looking for min depth or shortest path, we should immediately think of bfs.
def minDepth(self, root):
    if not root: return 0
    left = minDepth(root.left)
    right = minDepth(root.right)
    return min(left, right) + 1 if left and right else left+right+1



# p337 
# naive solution LTE
# this problem can be solved using dynamic programming. but this session is about dfs and recursion, check out the dp solution in dp file.
def rob(self, root):
    if not root: return 0
    val = 0
    if root.left:
        val += self.rob(root.left.left) + self.rob(root.left.right)

    if root.right:
        val += self.rob(root.right.left) + self.rob(root.right.right)

    return max(val + root.val, self.rob(root.left)+self.rob(root.right))
    


# 1026 
# key: maintain max node val so far and min node val so far, return max - min 
def maxAncestorDiff(root):
    def helper(r, mx, mi):
        if not r: return mx - mi
        # update max and min 
        mx = max(mx, r.val)
        mi = min(mi, r.val)
        # this makes the diff is from same ancestor
        return max(helper(r.left, mx, mi), helper(r.right, mx, mi))
    if not root: return 0
    # pass down the root's value
    return helper(root, root.val, root.val)


# 737
def areSentencesSimilarTwo(self, words1, words2, pairs):
    def dfs(s, t, visited, g):
        if t in g[s]: return True
        # avoid revisiting same node
        if s not in visited:
            visited.add(s)
            for w in g[s]:
                if dfs(w, t, visited, g): return True
        return False

    # build graph using dic
    graph = collections.defaultdict(set)
    for w1, w2 in pairs:
        graph[w1].add(w1)
        graph[w1].add(w2)
        graph[w2].add(w2)
        graph[w2].add(w1)
    
    if len(words1) != len(words2): return False
    for i, w in enumerate(words1):
        # if two words are the same continue
        if w == words2[i]: continue
        # word is not in the similarity graph return f 
        if w not in graph: return False
        if not dfs(w, words2[i], set(), graph): 
            return False 
    return True 


# p323
def countComponents(n, edges):
    def dfs(s, v, g):
        v[s] = True
        for i in g[s]:
            if not v[i]: 
                dfs(i, v, g)
         
    # build the graph
    graph = collections.defaultdict(set)
    for i, j in edges:
        graph[i].add(j)
        graph[j].add(i)
    cnt = 0
    visited = [False]*n
    for i in range(n):
        if not visited[i]:
            dfs(i, visited, graph)
            cnt += 1
    return cnt

    
# p261
# check if given edges form valid tree. 
# 1. no cicle
# 2. all nodes are reachable
# this is a undircted graph. 
# if the graph doesnt have cycle, parent indicator will overlap with current src, this can be used to terminate the process
def validTree(self, n, edges):
    def hascycle(g, src, visited, parent):
        visited[src] = True
        for n in g[src]:
            if visited[n] and n != parent or not visited[n] and hascycle(g, n, visited, src):
                return True 
        return False

    # build graph, because we dont need to quick access node, so we can actually use ajacency list to implement graph
    graph = collections.defaultdict(set)
    for i, j in edges:
        graph[i].add(j)
        graph[j].add(i)

    visited = [False]*n
    # if has cycle 
    if hascycle(graph, 0, visited, -1): 
        return False
    # if not reachable 
    for i in range(n):
        if not visited[i]: return False
    
    return True 


# p802
# there are two states in the problem: 1. safe, 2. unsafe
# write a function to check if all the paths of given source node can be at terminal and mark the nodes along the path safe or unsafe. as we know all the nodes inside circle are unsafe. 
# then we iterate through each node, check its state, store safe node in the result
# check out red black grey algorithm
def eventualSafeNodes(self, graph):
    states = ['safe', 'unsafe']
    def isSafe(g, src, mode):
        # base case, if a node's mode is safe return True else return false
        if mode[src]: return mode[src] == states[0]
        # initially set every node with unsafe mode
        mode[src] = states[1]
        # check circle
        for node in g[src]:
            if not isSafe(g, node, mode): return False 
        # if above code did not return, meaning src is safe, so set safe mode
        mode[src] = states[0]
        return True

    res = []
    if not graph: return res
    
    n = len(graph)
    mode = [None]*n
    for node in range(n):
        if isSafe(graph, node, mode):
            res.append(node)
    return res 


# p841
# core is check if the graph node are all reachable or connected.
# so we just check if size of all the unique keys used equal to total room exists. 
def canVisitAllRooms(self, rooms):
    visited = set()
    # start is the starting room key 
    def openRoom(start, rooms):
        # add current key in to the set 
        visited.add(start)
        for k in rooms[start]:
            if k not in visited:
                openRoom(k, rooms)

    openRoom(0, rooms)
    return len(visited) == len(rooms)


""""""""""""""""""""""""""""""""
""" BFS-Board and DFS and Dijkstra """
# bfs


def hasPath(maze, start, destination):
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    queue = [start]
    n, m = len(maze), len(maze[0])
    while queue:
        # every position in queue is where ball stops (border or 1)
        # using python list to simulate queue. pop front. unpacking
        i, j = queue.pop(0)
        # set visited positon value to -1
        maze[i][j] = -1
        if i == destination[0] and j == destination[1]:
            return True
        for x, y in directions:
            row = i + x
            col = j + y
            # loop until hit the wall or boarder
            while 0 <= row < n and 0 <= col < m and maze[row][col] != 1:
                row += x
                col += y
            # a step back
            row -= x
            col -= y
            if maze[row][col] == 0 and [row, col] not in queue:
                queue.append([row, col])
    return False


# from collections import heapq
def shortestDistance(maze, start, destination):
    pq = [(0, start[0], start[1])]
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    n, m = len(maze), len(maze[0])
    while pq:
        count, i, j = heapq.heappop(pq)
        if maze[i][j] == -1:
            continue
        maze[i][j] = -1
        if i == destination[0] and j == destination[1]:
            return count
        for x, y in directions:
            row = x + i
            col = y + j
            local = 1
            while 0 <= row < n and 0 <= col < m and maze[row][col] != 1:
                row += x
                col += y
                local += 1
            row -= x
            col -= y
            local -= 1
            # check out tuple comparison. basic idea is compare item by item lexigraphically.
            # maintian min heap, the shortest path always on the root.
            heapq.heappush(pq, (count+local, row, col))
    return -1

# bfs


def updateMatrix(matrix):
    queue = []
    n, m = len(matrix), len(matrix[0])
    for i in range(len(n)):
        for j in range(m):
            if matrix[i][j] != 0:
                # make sure all 1's get updated and added to queue
                matrix[i][j] = float('inf')
            else:
                queue.append((i, j))
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    while queue:
        cur_row, cur_col = queue.pop(0)
        for x, y in directions:
            row = cur_row + x
            col = cur_col + y
            if 0 <= row < n and 0 <= col < m:
                # if prev update is not the shortest, update it with shorter distance.
                if matrix[row][col] > matrix[cur_row][cur_col] + 1:
                    matrix[row][col] = matrix[cur_row][cur_col] + 1
                    queue.append((row, col))
    return matrix


def updateMatrix_dfs(matrix):
    n, m = len(matrix), len(matrix[0])
    for i in range(n):
        for j in range(m):
            if matrix[i][j] == 1:
                matrix[i][j] = float('inf')

    def dfs(matrix, i, j, n, m, dist):
        # matrix[i][j] can only be 0, inf, or some visited position
        # if matrix[i][j] is 0 or inf then matrix[i][j] < dist is false, when visited position less than dist, return meaning
        # right value is in place.
        if i < 0 or i >= n or j < 0 or j >= m or matrix[i][j] < dist:
            return
        matrix[i][j] = dist
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        for x, y in directions:
            dfs(matrix, i+x, j+y, n, m, dist+1)

    for i in range(n):
        for j in range(m):
            if matrix[i][j] == 0:
                dfs(matrix, i, j, n, m, 0)
    return matrix


def wallsAndGates(rooms):
    if not rooms:
        return rooms
    queue = []
    n, m = len(rooms), len(rooms[0])
    for i in range(n):
        for j in range(m):
            if rooms[i][j] == 0:
                queue.append((i, j))

    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    while queue:
        cur_row, cur_col = queue.pop(0)
        for x, y in directions:
            row = cur_row + x
            col = cur_col + y
            if 0 <= row < n and 0 <= col < m and rooms[row][col] != -1:
                if rooms[row][col] > rooms[cur_row][cur_col] + 1:
                    rooms[row][col] = rooms[cur_row][cur_col] + 1
                    queue.append((row, col))
    return rooms


def floodFill(image, sr, sc, newColor):
    if image[sr][sc] == newColor:
        return image

    def dfs(image, i, j, old_color, new_color):
        if i < 0 or i >= len(image) or j < 0 or j >= len(image[0]) or image[i][j] != old_color:
            return
        image[i][j] = new_color
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        for x, y in directions:
            dfs(image, i+x, j+y, old_color, new_color)

    dfs(image, sr, sc, image[sr][sc], newColor)
    return image


# dfs + memoization
def longestIncreasingPath(matrix):
    def dfs(matrix, i, j, n, m, cache):
        if cache[i][j]:
            return cache[i][j]
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        leng = 1
        for x, y in directions:
            x += i
            y += j
            if x < 0 or y < 0 or x >= n or y >= m or matrix[i][j] >= matrix[x][y]:
                continue
            leng = max(leng, dfs(matrix, x, y, n, m, cache)+1)
        cache[i][j] = leng
        return cache[i][j]

    if not matrix:
        return 0
    n, m, ans = len(matrix), len(matrix[0]), 1
    cache = [[0 for _ in range(m)] for _ in range(n)]

    for i in range(n):
        for j in range(m):
            ans = max(ans, dfs(matrix, i, j, n, m, cache))
    return ans


def pacificAtlantic(matrix):
    def dfs(matrix, i, j, n, m, visited):
            visited[i][j] = True
            directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
            for x, y in directions:
                x += i
                y += j
                if x < 0 or y < 0 or x >= n or y >= m or visited[x][y] or matrix[i][j] > matrix[x][y]:
                    continue
                dfs(matrix, x, y, n, m, visited)

    if not matrix:
        return []
    n, m, ans = len(matrix), len(matrix[0]), []
    pa_visited = [[False for _ in range(m)] for _ in range(n)]
    at_visited = [[False for _ in range(m)] for _ in range(n)]
    # if inner cells can flow to first colunm and flow to last column
    for i in range(n):
        dfs(matrix, i, 0, n, m, pa_visited)
        dfs(matrix, i, m-1, n, m, at_visited)
    # if inner cells can flow to first row and flow to last row
    for j in range(m):
        dfs(matrix, 0, j, n, m, pa_visited)
        dfs(matrix, n-1, j, n, m, at_visited)

    for i in range(n):
        for j in range(m):
            if pa_visited[i][j] and at_visited[i][j]:
                ans.append([i, j])
    return ans


# 743 Dijkstra algorithm see TCRC
def networkDelayTime(times, N, K):
    # use defaultdict with list factory meaning the value of given key will be store in a list
    # [(0, k)] (estimate, node)
    # s is for storing shortest path to each node
    pq, s, adj = [(0, K)], {}, collections.defaultdict(list)
    for u, v, w in times:
        # representing edges
        adj[u].append((w, v))

    while pq:
        time, node = heapq.heappop(pq)
        if node not in s:
            s[node] = time  # shortest path to this node from source
            for w, v in adj[node]:
                heapq.heappush(pq, (w+time, v))
    return max(s.values()) if len(s) == N else -1

# 787


def findCheapestPrice(n, flights, src, dst, K):
    pq, edges = [(0, src, 0)], collections.defaultdict(list)
    for u, v, w in flights:
        edges[u].append((w, v))

    while pq:
        price, node, stops = heapq.heappop(pq)
        if node == dst:
            return price
        if stops <= K:
            # add all current node's neibours to the queue, basically like bfs(use normal queue) but with priorityqueue instead
            for p, n in edges[node]:
                heapq.heappush(pq, ((p+price), n, stops+1))
    return -1
