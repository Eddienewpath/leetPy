# 501
def findMode_extra_space(self, root):
    if not root: return []
    dic = {}
    self.dfs(root, dic)

    max_freq = max(dic.values())

    res = []
    for v, f in dic.items():
        if f == max_freq: res.append(v)
    return res


def dfs(self, root, dic):
    if not root:
        return 
    self.dfs(root.left, dic)
    dic[root.val] = dic.get(root.val, 0) + 1 
    self.dfs(root.right, dic)


""" follow up implement O(1) space solution """
def findMode(self, root):
    pass 


class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

# 108
def sortedArrayToBST(self, nums):
    if nums:
        mid = len(nums)//2
        root = TreeNode(nums[mid])

        root.left = self.sortedArrayToBST(nums[:mid])
        root.right = self.sortedArrayToBST(nums[mid+1:])
        return root

# 530
""" key: BST doing a in order traversal will get sorted array. and the min absolute difference should exist in between two conitnuous number in the array
thus, we only need to maintain a prev variable to compare with current node and update the min_diff along the way """
class Solution(object):
    def __init__(self):
        self.min_diff = float('inf')
        self.prev = None

    # simply do a in-order traversal and return updated min_diff
    def getMinimumDifference(self, root):
        if not root:
            return self.min_diff

        self.getMinimumDifference(root.left)

        if self.prev:
            self.min_diff = min(self.min_diff, root.val - self.prev.val)

        self.prev = root

        self.getMinimumDifference(root.right)
        return self.min_diff



# 700
def searchBST(self, root, val):
    if not root: return     
    if root.val < val:
        return self.searchBST(root.right, val)
    elif root.val > val:
        return self.searchBST(root.left, val)
    else:
        return root
        


# 450
""" tip: when doing recursive problems, clearly define the what the function does and appy to each stack call see if it logically make sense """
# function deletes the node with key value, and return a new node that will maintain the BST property
""" copy_value implementation """
def deleteNode(self, root, key):
    if not root:
        return
    if root.val > key:
        root.left = self.deleteNode(root.left, key)
    elif root.val < key:
        root.right = self.deleteNode(root.right, key)
    else:
        if not root.left:
            return root.right
        elif not root.right:
            return root.left
        else:
            """ easiest way to maintain a BST when parent is deleted, is to find the min value on the right subtree and 
            put it at the parent position and remove it from its old position, thus both sides maintain bst 
            tip: all nodes on the right side are greater than the nodes on the left side, even the smallest on the right side.
             """
            min_node = self.find_min(root.right)
            #assign smallest value to the root and delete the node with smallest value will maintain left and right to be valid bst.
            root.val = min_node.val
            #this line will maintain the root.right to be a bst
            root.right = self.deleteNode(root.right, min_node.val)
    return root


# find the subtree on with min value
def find_min(self, root):
    while root.left:
        root = root.left
    return root


""" above implementaion is simply copy the node value, does not really delete the node, thus below version will implelent it using rotation"""
def deleteNode_rotate(self, root, key):
    pass 



# 98 
# recursive solution
# preorder or postorder traversal both will work
def isValidBST(self, root):
    return self.isValidBST_helper(root, float('-inf'), float('inf'))


def isValidBST_helper(self, root, mi, mx):
    if not root: return True
    if root.val <= mi or root.val >= mx: return False

    left = self.isValidBST_helper(root.left, mi, root.val)
    right = self.isValidBST_helper(root.right, root.val, mx)

    return left and right

# iterative solution
""" using in-order traversal iterative way
because in-order traversal for bst will produce sorted array, we only need to maintain a prev, so if current node value is 
less and equal to prev, return false """

def isValidBST(self, root):
    stack, prev = [], None
    while stack or root:
        while root:
            stack.append(root)
            root = root.left
        root = stack.pop()
        """ tip: use is None or is not None as condition to avoid bugs """
        if prev is not None and root.val <= prev:
            return False
        prev = root.val
        root = root.right
    return True


# 230 
def kthSmallest(self, root, k):
    stack = []
    while stack or root:
        while root:
            stack.append(root)
            root = root.left
        
        root = stack.pop()
        k -= 1
        if k == 0: return root.val 
        root = root.right
    return -1
        


# 701 
# insert given node into the BST and return the modified BST 
def insertIntoBST(self, root, val):
    if not root: return TreeNode(val)
    if root.val > val:
        # insert into the leftsubtree and return the modified left subtree. 
        root.left = self.insertIntoBST(root.left, val)
        
    elif root.val < val:
        root.right = self.insertIntoBST(root.right, val)   
    return root


# 99 
# in-order iterative
def recoverTree_iter(self, r):
    stack, prev = [], None
    one, two = None, None
    while stack or r:
        while r:
            stack.append(r)
            r = r.left
        
        r = stack.pop()
        """ this part is a classic probelm: identify two swapped elements in sorted array """
        if prev and prev.val >= r.val:
            one = r
            if not two:
                two = prev
            else:
                break   
        """ end """
        prev = r
        r = r.right
    one.val, two.val = two.val, one.val

""" implement morris traversal in pace """
def recoverTree(self, r):
    pass 



# 968
def minCameraCover(self, root):
    """
    :type root: TreeNode
    :rtype: int
    """
    pass 
