
# 1
# /\
# 2 2   
# /\/\
# 3443

# if a tree is symetric, its left and right trees should be symetric to each other and their root should be equal
def isSymmetric(root):
    return helper   (root, root) 

# return true if two trees are symetric
def helper(r1, r2):
    if not r1 and not r2: return True 
    if not r1 or not r2: return False
   
    return r1.val == r2.val and helper(r1.left, r2.right) and helper(r1.right, r2.left)


# two tree nodes value must be the same to be symetric
def iter_isSymmetric(root):
    if not root: return True
    stack = [root.left, root.right]
    while stack: 
        t1 = stack.pop()
        t2 = stack.pop()
        if not t1 and not t2: continue
        if not t1 or not t2: return False
        if t1.val != t2.val: return False
        stack.append(t1.left)
        stack.append(t2.right)
        stack.append(t1.right)
        stack.append(t2.left)

    return True
        
