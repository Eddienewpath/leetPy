def flatten(root):
    def helper(r):
        if not r: return None
        suc = helper(r.left)
        first = helper(r.right)
        if suc:
            r.left = None
            r.right = suc
            while suc.right:
                suc = suc.right
            suc.right = first
        return r
    helper(root)


def flatten_iter(root):
    if not root: return root
    # stores direct subtree root
    stack = [root] 
    while stack:
        t = stack.pop()
        if t.right: 
            stack.append(t.right)

        if t.left:
            stack.append(t.left)

        if stack:
            t.right = stack[-1]

        t.left = None
        
        
        
#     1
#    / \
#   2   5
#  / \   \
# 3   4   6



# put tmp = r.right , r.right = r.left, r.left = None r.left.right = tmp
# make left tree become the right tree and orignal right tree atach to the endo


# def flatten(root):
#     def helper(r):
#         if not r:
#             return None
#         suc = helper(r.left)
#         r.left = None
#         first = helper(r.right)
#         r.right = first
#         if first:
#             pre = first
#             while pre.right:
#                 pre = pre.right
#             pre.right = suc
#         return r
#     helper(root)
