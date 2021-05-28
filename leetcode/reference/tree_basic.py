

""""""""""""""""""""""""""""""""
""" Tree basic"""


class TreeNode(object):
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None


# class ListNode(object):
#     def __init__(self, val):
#         self.val = val
#         self.next = None
root = TreeNode(1)
root.left = TreeNode(2)
root.right = TreeNode(3)
root.left.left = TreeNode(4)
root.left.right = TreeNode(5)
root.right.left = TreeNode(6)
root.right.right = TreeNode(7)

# output: 4251637


def in_order(r):
    stack = []
    res = []
    while r or stack:
        while r:
            stack.append(r)
            r = r.left
        if stack:
            r = stack.pop()
            res.append(r.val)
            r = r.right
    return res
# print(in_order(root))

# push all left side into stack all at the same time


def pre_order_ii(root):
    stack = []
    res = []
    while stack or root:
        while root:
            res.append(root.val)
            stack.append(root)
            root = root.left
        root = stack.pop()
        root = root.right
    return res


# output: 1245367
# push only children of the root to the stack and the order is matter here, we need to push right then left, thus we can
# access left first for the next pop.
def pre_order_i(r):
    stack = [r]
    res = []
    while stack:
        top = stack.pop()
        res.append(top.val)
        if top.right:
            stack.append(top.right)
        if top.left:
            stack.append(top.left)
    return res


print(pre_order_ii(root), pre_order_i(root))

#  1
# /\
# 2 3
# /\
# 4 5

# 4526731


def post_order(r):
    stack, res = [r], []
    while stack:
        top = stack.pop()
        res.append(top.val)
        if top.left:
            stack.append(top.left)
        if top.right:
            stack.append(top.right)

    return res[::-1]
# print(post_order(root))


# head = ListNode(1)
# head.next = ListNode(2)
# head.next.next = ListNode(3)
# head.next.next.next = ListNode(4)


# iter
# 1-2-3-4
def reverseLL_iter(head):
    pre, cur = None, None
    while head:
        pre = head
        head = head.next
        pre.next = cur
        cur = pre
    return cur

# head = reverseLL_iter(head)
# while head:
#     print(head.val)
#     head = head.next


def reverseLL_recur(head):
    if not head.next:
        return head
    new_head = reverseLL_recur(head.next)
    head.next.next = head
    head.next = None
    return new_head


# head = reverseLL_recur(head)
# while head:
#     print(head.val)
#     head = head.next


def isSymetric_recur(r):
    def helper(r1, r2):
        if not r1 and not r2:
            return True
        if not r1 or not r2:
            return False
        return r1.val == r2.val and helper(r1.left, r2.right) and helper(r1.right, r2.left)
    return helper(r, r)

#   1
#  /\
# 2  2
# /\ /\
# 1 2 2 1


def isSymetric_iter(r):
    def helper(r1, r2):
        stack = [r2, r1]
        while stack:
            r1 = stack.pop()
            r2 = stack.pop()
            # the order of following two line can be swapped. if swappped will return false when both are null nodes
            if not r1 and not r2:
                continue
            if not r1 or not r2:
                return False
            if r1.val != r2.val:
                return False
            stack.append(r2.right)
            stack.append(r1.left)
            stack.append(r2.left)
            stack.append(r1.right)
        return True
    return helper(r, r)


# print(isSymetric_iter(root))
# print(isSymetric_recur(root))
