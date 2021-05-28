
# 543 
# very slow to-down preorder traversal, for every node need to calculate the height of left and right subtree
def diameterOfBinaryTree(self, root):
    if not root: return 0
    left = self.diameterOfBinaryTree(root.left)
    right = self.diameterOfBinaryTree(root.right)
    thru_root = self.height(root.left) + self.height(root.right)
    return max(thru_root, left, right)

# this is an postorder traversal, bottom up
def height(self, r):
    if not r: return 0 
    return max(self.height(r.left), self.height(r.right)) + 1


# some optimalization: using dictionary to precalculate the heights of each node
""" tip: number of nodes in the subtrees equal to the total edges count """
def diameterOfBinaryTree_dic(self, root):
    dic = {}
    self.height(root, dic)
    return self.helper(root, dic)


def helper(self,root, dic):
    if not root: return 0 
    left = self.helper(root.left, dic)
    right = self.helper(root.right, dic)
    left_height = dic[root.left] if root.left else 0 
    right_height = dic[root.right] if root.right else 0
    thru_root = left_height + right_height
    return max(thru_root, left, right)

def height(self, r, dic):
    if not r: return 0 
    h = max(self.height(r.left, dic), self.height(r.right, dic)) + 1
    dic[r] = h
    return h 


# more optimization: maintain a global variable to keep a max distance thru root.
class Solution(object):
    def __init__(self):
        self.ans = 0

    def diameterOfBinaryTree(self, root):
        self.height(root)
        return self.ans

    def height(self, r):
        if not r:
            return 0
        left, right = self.height(r.left), self.height(r.right)
        # left + right is total number of nodes in the subtrees , which is equal to the total edge count going thru root.
        self.ans = max(self.ans, left+right)
        return max(left, right) + 1


# 124
class Solution(object):
    def __init__(self):
        self.res = float('-inf')

    def maxPathSum(self, root):
        self.maxPathSumFromRoot(root)
        return self.res

    # return max path sum from given root node to some node in the tree
    """ find the path that has the largest sum from root to some node in the tree and return the sum """
    def maxPathSumFromRoot(self, root):
        if not root:
            return float('-inf')
        left = self.maxPathSumFromRoot(root.left)
        right = self.maxPathSumFromRoot(root.right)
        # this updates the largest path sum along the way back to the top roots
        '''the max path sum must be some path that go to or go thru some root in the tree'''
        self.res = max(self.res, max(left, 0) + max(right, 0) + root.val)
        # the returned is just one path starting from current root
        return max(left, right, 0) + root.val




""" tip: post-order our single stack call logic is better apply to the last stack that close to the base case because we are work from the bottom up """

# 687 similar to above problem
""" tip: when ask about length of path, think of number of nodes instead. one node we see it as no path coz two nodes form a path """
class Solution(object):
    def __init__(self):
        self.longest = 0

    def longestUnivaluePath(self, root):
        self.longestPathHelper(root)
        return self.longest

    def longestPathHelper(self, root):
        if not root:
            return 0
        left, right = self.longestPathHelper(root.left), self.longestPathHelper(root.right)
        # number of nodes on the longest unival-path on the left, and right
        # root.left.val == root.val form a valid edge will contribute to path
        left = left + 1 if root.left and root.left.val == root.val else 0
        right = right + 1 if root.right and root.right.val == root.val else 0
        self.longest = max(self.longest, left + right)
        return max(left, right)



# 235 
# if p or q is equal to value of root, that is the LCA
def lowestCommonAncestor(self, root, p, q):
    if not root:
        return
    if p.val < root.val and q.val < root.val:
        return self.lowestCommonAncestor(root.left, p, q)
    elif p.val > root.val and q.val > root.val:
        return self.lowestCommonAncestor(root.right, p, q)
    else:
        return root



# 1325 
""" key is to recognize that we need to traverse from bottom to top, thus we use post-order """
def removeLeafNodes(self, root, target):
    if not root:
        return None
    """ dont put checking condition here , this will not work because it becomes preorder traversal"""
    root.left = self.removeLeafNodes(root.left, target)
    root.right = self.removeLeafNodes(root.right, target)
    # post-order, process the root at last. 
    if not root.left and not root.right and root.val == target:
        return None
    else:
        return root


# 129 
class Solution(object):
    def __init__(self):
        self.total = 0

    def sumNumbers(self, root):
        self.build_path(root, 0)
        return self.total

    def build_path(self, root, val):
        if not root:
            return
        if not root.left and not root.right:
            val = val*10+root.val
            self.total += val
            return

        self.build_path(root.left, val*10+root.val)
        self.build_path(root.right, val*10+root.val)



# 337 
# TLE 
def rob_TLE(self, root):
    if not root: return 0
    if root: 
        left = self.rob(root.left)
        right = self.rob(root.right)
        
    total = 0
    if root.left:
        total += self.rob(root.left.left)
        total += self.rob(root.left.right)
    if root.right:
        total += self.rob(root.right.left)
        total += self.rob(root.right.right)
    total += root.val
    return max(left + right, total)

# hashmap version to pass the test
def rob(self, root):
    dic = {}
    return self.helper(root, dic)
    
    
def helper(self, root, dic):
    if not root: return 0
    if root in dic: return dic[root]
    if root: 
        left = self.helper(root.left, dic)
        right = self.helper(root.right, dic)

    total = 0
    if root.left:
        total += self.helper(root.left.left, dic)
        total += self.helper(root.left.right,dic)
    if root.right:
        total += self.helper(root.right.left,dic)
        total += self.helper(root.right.right,dic)
    total += root.val
    mx = max(left + right, total)
    dic[root] = mx
    return mx

""" DP version implementing later """
def rob(self, root):
    pass 



# 979
""" 
post order traversal(bottom up). 
we still abstract out the tree in to root and left and right subtree. 
thus we need to balance out left and right subtrees, and because the number of nodes and number of coins are equal
so when left and right subtrees are balanced, the root tree is balanced
and the flow between left subtree and root node and flow between right subtree and the root node,
their sum will be the total steps needed to balance the tree. 
 """
class Solution(object):
    def __init__(self):
        self.flows = 0

    def distributeCoins(self, root):
        self.balance(root)
        return self.flows

    def balance(self, root):
        if not root:
            return 0
        # give out or receive in
        left = self.balance(root.left)
        right = self.balance(root.right)
        self.flows += abs(left) + abs(right)
        return left + right + root.val - 1



