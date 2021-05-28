class Node(object):
    def __init__(self, val=None, children=None):
        self.val = val
        self.children = children

class Solution(object):
    def levelOrder(self, root):
        if not root: return []
        import collections
        queue = collections.deque()
        res = []
        queue.append(root)
        while queue:
            size = len(queue)
            tmp = []
            while size:
                front = queue.popleft()
                for c in front.children: 
                    queue.append(c)
                tmp.append(front.val)
                size -= 1
            res.append(tmp)
        return res


def isSymmetric(self, root):
        return self.helper(root, root)

def helper(self, r1, r2):
    # the order of two condition below cannot be reversed
    if not r1 and not r2: return True
    if not r1 or not r2: return False
    return r1.val == r2.val and self.helper(r1.right, r2.left) and self.helper(r1.left, r2.right)

def isSymmetric_iter(self, root):
    import collections
    # intuition: mirror of itself, we can actually compare two tree at the same time
    queue = collections.deque([root, root])
    while queue:
        n1 = queue.popleft()
        n2 = queue.popleft()
        if not n1 and not n2: continue
        if not n1 or not n2: return False
        if n1.val != n2.val: return False
        queue.append(n1.right)
        queue.append(n2.left)
        queue.append(n1.left)
        queue.append(n2.right)
    return True
            

# dfs 
def maxDepth(self, root):
    if not root:
        return 0
    return max(self.maxDepth(root.left), self.maxDepth(root.right)) + 1

# find number of level will be the depth of the tree
def maxDepth_bfs(self, root):
    if not root: return 0 
    import collections
    queue = collections.deque()
    queue.append(root)
    level = 0
    while queue:
        size = len(queue)
        while size:
            front = queue.popleft()
            if front.left: 
                queue.append(front.left)
            if front.right:
                queue.append(front.right)
            size -= 1 
        level += 1
    return level


# very slow coz repeatedly calculate the height of nodes from bottom up 
def isBalanced(self, root):
    if not root: return True 
    return self.isBalanced(root.left) and self.isBalanced(root.right) and abs(self.height(root.left) - self.height(root.right)) <= 1

def height(self, root):
    if not root: return 0 
    return max(self.height(root.left), self.height(root.right)) + 1
    
#  faster version: calculate the height and at the same time check if subtree is 
# follow the balanced tree definition by using a signal -1 to indicate that. 
def isBalanced(self, root):
    def check(root):
        if not root:
            return 0
        left = check(root.left)
        right = check(root.right)
        if left == -1 or right == -1 or abs(left - right) > 1:
            return -1
        return max(left, right) + 1
    return check(root) != -1


def minDepth(self, root):
        if not root:
            return 0
        left = self.minDepth(root.left)
        right = self.minDepth(root.right)
        if left == 0:
            return right + 1
        if right == 0:
            return left + 1
        return min(left, right) + 1




def deepestLeavesSum(self, root):
    if not root: return 0
    queue = collections.deque([root])
    res = None
    while queue:
        size = len(queue)
        res = 0
        while size:
            front = queue.popleft()
            res += front.val
            if front.left: 
                queue.append(front.left) 
            if front.right:
                queue.append(front.right)
            size -= 1
    return res
        
    
# using set
def isUnivalTree(self, root):
    def helper(root, path):
        if not root: return
        path.add(root.val)
        helper(root.left, path)
        helper(root.right, path)
        
    path = set()
    helper(root, path)
    return len(path) == 1

# O(1) space
def isUnivalTree_(self, root):
    if not root:
        return True
    if root.left:
        if root.val != root.left.val:
            return False
    if root.right:
        if root.val != root.right.val:
            return False
    return self.isUnivalTree(root.left) and self.isUnivalTree(root.right)

