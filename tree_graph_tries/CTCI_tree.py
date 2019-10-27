class TreeNode(object):
    def __init__(self, val):
        self.val = val
        self.left = None 
        self.right = None
        self.parent = None



# given sorted array, to build min height bst
# two steps: first find the root, second connect the left subtree and right subtree to the root and return the root
# [1,2,3,4,5]
# problem: 4.2
def build_min_tree(arr):
    if not arr: return None
    def helper(arr, start, end):
        if start > end: return
        mid = (start + end) // 2
        root = TreeNode(arr[mid])
        root.left = helper(arr, start, mid-1)
        root.right = helper(arr, mid+1, end)
        return root
    return helper(arr, 0, len(arr)-1)


def build_min_tree_iter(arr):
    if not arr: return None
    root = TreeNode(-1)
    node_stack = [root]
    start_stack = [0]
    end_stack = [len(arr)-1]

    while node_stack: 
        r = node_stack.pop()
        start = start_stack.pop()
        end = end_stack.pop()
        mid = (start+end) // 2
        r.val = arr[mid]
        if start <= mid-1: 
            r.left = TreeNode(-1)
            node_stack.append(r.left)
            start_stack.append(start)
            end_stack.append(mid-1)
        
        if mid+1 <= end: 
            r.right = TreeNode(-1)
            node_stack.append(r.right)
            start_stack.append(mid+1)
            end_stack.append(end)
    return root
        

# r = build_min_tree_iter([1,2,3,4,5])
# # inorder to display result
# stack = []
# while r or stack: 
#     while r: 
#         stack.append(r)
#         r = r.left
#     if stack: 
#         r = stack.pop()
#         print(r.val)
#         r = r.right




class ListNode(object):
    def __init__(self, val):
        self.val = val 
        self.next = None



# convert each level nodes into a linked list and return the lists
# problem: 4.3
# dfs


def level_lists_dfs(r):
    if not r: return []
    ans_list = []
    def helper(r, level, ans):
        if not r: return
        if level >= len(ans): 
            head = cur = ListNode(-1)
            ans.append([head, cur])
        
        ans[level][1].next = ListNode(r.val)
        ans[level][1] = ans[level][1].next
        helper(r.left, level+1, ans)
        helper(r.right, level+1, ans)
    helper(r, 0, ans_list)
    res = []
    for arr in ans_list:
        res.append(arr[0].next)
    return res



import collections
def level_lists_bfs(r):
    if not r: return []
    res = []
    queue = collections.deque()
    queue.append(r)
    while queue: 
        size = len(queue)
        dummy = ListNode(-1)
        for i in range(size):
            front = queue.popleft()
            dummy.next = ListNode(front.val)
            if i == 0: 
                res.append(dummy.next)
            dummy = dummy.next
            if front.left: 
                queue.append(front.left)
            if front.right: 
                queue.append(front.right)
    return res


# heads = level_lists_bfs(None)
# for h in heads:
#     print('level')
#     while h:
#         print(h.val)
#         h = h.next


# check if a binary tree is balanced
# problem: 4.4
def isBalanced(r):
    # return -1 if tree is not balanced and return height of the three if tree is balanced
    def helper(r):
        if not r: return 0 
        left = helper(r.left)
        right = helper(r.right)
        if left == -1 or right == -1: return -1
        if abs(left - right) > 1: return -1
        return max(left, right) + 1
    return helper(r) != -1

# print(isBalanced(root))



# if left and right subtree is bst and left(r) < r < right(r) 
# ///// be careful, following is wrong becuase it does not guruntee all the left nodes values are less than root val
# ///// or all the right nodes val is greater than root.val
# if not r or (not r.left and not r.right):
#     return True
#     if not r.left:
#         return r.right.val > r.val and isBST(r.right)
#     if not r.right:
#         return r.left.val <= r.val and isBST(r.left)
#     else:
#         return r.left.val <= r.val < r.right.val and isBST(r.left) and isBST(r.right)
# problem: 4.5

# preorder solution
def isBST_arr(r):
    arr = []
    def helper(r):
        if not r: return
        helper(r.left)
        arr.append(r.val)
        helper(r.right)
    helper(r)
    for i in range(1, len(arr)):
        if arr[i] < arr[i-1]: return False
    return True


# class someclass(object):
def isBST_var(r):
    last = None 
    def helper(r):
        # nonlocal explanation here : https://eli.thegreenplace.net/2011/05/15/understanding-unboundlocalerror-in-python 
        nonlocal last
        if not r: 
            return True
        if not helper(r.left): 
            return False
        if last != None and r.val < last: 
            return False 
        last = r.val
        if not helper(r.left): return False
        return True
    return helper(r)


# using min and max, if branch left, r.val is the max, if branch right, r.val is the min
def isBST_recur(r):
    def helper(r, min_val, max_val): 
        if not r: return True
        # check if current root val is in range
        if (min_val != None and r.val <= min_val) or (max_val != None and r.val >= max_val): return False
        # check if both subtrees are bst
        if not helper(r.left, min_val, r.val) or not helper(r.right, r.val, max_val): return False

        return True 
    return helper(r, None, None)

# print(isBST_recur(root))  

#   3
#  /\
#  2 6
#  /\/\
# 1 4 5 7



root = TreeNode(3)
root.parent = None
node2 = TreeNode(2)
node2.parent = root
root.left = node2
node1 = TreeNode(1)
node1.parent = node2
root.left.left = node1
node6 = TreeNode(6)
node6.parent = root
root.right = node6
node4 = TreeNode(4)
node4.parent = node2
node7 = TreeNode(7)
node7.parent = node6
root.right.right = node7
root.left.right = node4
node5 = TreeNode(5)
node5.parent = node6 
root.right.left = node5
# each node has a link to its parent
# find given node's next node (in order traversal next node)
# if node is leftmost, its successor is its parent, if node is right most node 

def successor(node):
    if not node: return None
    if not node.right and not node.parent: return None
    if not node.right:
        if node.parent.left == node:
            return node.parent
        else:
            while node.parent and node.parent.right == node:
                node = node.parent
            return node.parent
    else:
        r = node.right
        while r.left:
            r = r.left
        return r
    

# print('None' if successor(node5) == None else successor(node5).val)


# problem:4.8
# binary tree 
# return CA of two nodes
# algorithm: find p and q in r's subtrees and if they in different tree, return their CA
def commonAncestor(r, p, q):
    if not r: return None
    if p == r == q: return r
    # find p and q 
    if r.val == p.val: return p
    if r.val == q.val: return q
    left = commonAncestor(r.left, p, q)
    right = commonAncestor(r.right, p, q)
    # if two nodes are found in two subtree, return their current root
    if left and right: return r
    # if only one is found, return that one to pre level
    if left: return left
    if right: return right
    else: return None
    

# print(commonAncestor(root, node5, node7).val)



    #   3
    #  /\
    #  2 6
    #  /\/\
    # 1 4 5 7
    # if a node is in subtree of the other node, return the other node
    # 
    # if q or q is null, return q or p
    # if q and p are null, return null
    # q
    # /\
    # n n


# problem: 4.10 
# t1 is much bigger than t2, check if t2 is subtree of t1. binary tree. 
# there exists a node, t, in t1 st. t has identical structue and val as t2
    #   3
    #  /\
    #  2 6
    #  /\/\
    # 1 4 5 7
root2 = TreeNode(6)
root2.left = TreeNode(5)
root2.right = TreeNode(7)

def check_subtree(t1, t2):
    if not t1 and not t2: return True

    def is_identical(t1, t2):
        if not t1 and not t2:
            return True
        if not t1 or not t2:
            return False
        return t1.val == t2.val and is_identical(t1.left, t2.left) and is_identical(t1.right, t2.right)

    def find_root(r, alist, tar):
        if not r:
            return
        if r.val == tar:
            alist.append(r)
        find_root(r.left, alist, tar)
        find_root(r.right, alist, tar)

    matched_roots = []
    find_root(t1, matched_roots, t2.val)
    
    for t in matched_roots:
        if is_identical(t, t2): return True

    return False 

print(check_subtree(root, root2))




