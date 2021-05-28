
# 437 
""" brute force solution try all the nodes in the tree to find all the paths """
class Solution(object):
    def __init__(self):
        self.count = 0

    def pathSum(self, root, sum):
        self.helper(root, sum)
        return self.count

    def helper(self, root, s):
        if not root:
            return
        self.num_path(root, s)
        self.helper(root.left, s)
        self.helper(root.right, s)

    def num_path(self, root, s):
        if not root:
            return
        """ dont return the stack frame coz it will have edge case, for example [1,-2,-3,1,3,-2,null,-1], -1 
        for this case, it will not include path [1 -> -2 -> 1 -> -1] 
        if not returning, the recurive call will go deeper to include this path """
        if s == root.val:
            self.count += 1

        self.num_path(root.left, s - root.val)
        self.num_path(root.right, s - root.val)


""" prefix sum solution """
def pathSum(self, root, sum):
    pass 


# 235 
""" single stack frame logic: go find the p or q node from left or right side of tree, and return the results(references) to p or q or not found
if result return inlcude both references of p or q meaning current root is the LCA else if p is found, p is the parent of q return p else return q """
def lowestCommonAncestor(self, root, p, q):
    if not root: return None
    # similar to path compression, return the found root reference to the above level
    if root in (p, q): return root
    """ recursion, each stack frame will maintain or remember some states and waiting for rest of recursion code to return and then combine the result to solve the problem """
    left = self.lowestCommonAncestor(root.left, p, q)
    right = self.lowestCommonAncestor(root.right, p, q)

    if left and right:
        return root
    elif not right:
        return left
    else:
        return right

# iterative solution
def lowestCommonAncestor_iter(self, root, p, q):
    pass 



# 669 
# define what this function will do. 
# this function will trim the BST  and return the new root of the tree.
def trimBST(self, root, L, R):
    if not root: return 
    """ basically this is preorder traversal. 
    check if root val is in range 
    if root is not in range, starts function at its children using the BST characteristics """
    # if root is not in range, we only need to check either left or right child
    if root.val < L: return self.trimBST(root.right, L, R)
    if root.val > R: return self.trimBST(root.left, L, R)
    # if root is in range, we need to trim both children
    root.left = self.trimBST(root.left, L, R)
    root.right = self.trimBST(root.right, L, R)
    return root


# 814 
""" pre-order version """
def pruneTree_pre(self, root):
    if not root: return 
    if self.has_one(root): 
        root.left = self.pruneTree(root.left)
        root.right = self.pruneTree(root.right)
    else: 
        root = None
    return root 
    
    
def has_one(self, r):
    if not r: return False
    return r.val == 1 or self.has_one(r.left) or self.has_one(r.right)
    
""" post order, this will avoid using second helper function to check if subtree including 1's or not """
""" single stack logic: prune left subtree and prune right subtree, check if 1's in left or right subtree and if current root value is 1 or not
if no 1 found return None else return current root """
def pruneTree_post(self, root):
    if not root: return None 
    root.left = self.pruneTree(root.left)
    root.right = self.prunTree(root.right)
    if not root.left and not root.right and not root.val: return None
    return root


# 872
def leafSimilar(self, root1, root2):
    #compare if two nodes have same leaves
    return self.find_leaf(root1) == self.find_leaf(root2)

#return a list of leaves
def find_leaf(self, r):
    # if null tree, has no leaf, return empty list 
    if not r: return []
    # if a tree node has not children, that node is the leaf, add to the list and return
    if not r.left and not r.right: return [r.val]
    #combine leaves from left and right subtrees
    return self.find_leaf(r.left) + self.find_leaf(r.right)
    
