class TreeNode:
    def __init__(self, val):
        self.left = None
        self.right = None
        self.val = val

#    4
#  /  \
#  2   6
# /\  /\
# 1 3 5 7


# [1,2,3,4,5,6,7]
def in_order(r):
    res = []
    stack = []
    # when stack is empty and cur node is null, meaning our process is done.
    while stack or r:
        while r: 
            stack.append(r)
            r = r.left
        if stack: 
            r = stack.pop()
            res.append(r.val)
            r = r.right
    return res


# [4,2,1,3,6,5,7]
def pre_order(r):
    res = []
    stack = []
    while stack or r: 
        while r: 
            res.append(r.val)
            stack.append(r)
            r = r.left
        if stack: 
            r = stack.pop()
            r = r.right
    return res 


#    4
#  /  \
#  2   6
# /\  /\
# 1 3 5 7

# [1,3,2,5,7,6,4]
# bottom -> top and left -> right
# intuition: top -> bottom and right -> left s
def post_order(r):
    if not r: return []
    res, stack = [], [r]
    while stack: 
        top = stack.pop()
        res.append(top.val)
        if top.left: 
            stack.append(top.left)
        if top.right:
            stack.append(top.right)

    return res[::-1]

def post_order_ii(r):
    res, stack  = [], []
    while stack or r: 
        while r:
            res.append(r.val)
            stack.append(r)
            r = r.right
        if stack: 
            r = stack.pop() 
            r = r.left
    return res[::-1]    



root = TreeNode(4)
left2 = TreeNode(2)
right6 = TreeNode(6)
left1 = TreeNode(1)
right3 = TreeNode(3)
left5 = TreeNode(5)
right7 = TreeNode(7)


root.left = left2
root.right = right6
left2.left = left1
left2.right = right3
right6.left = left5
right6.right = right7

print(post_order_ii(root))

