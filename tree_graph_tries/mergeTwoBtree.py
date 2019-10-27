def mergeTrees(t1, t2):
    return helper(t1, t2) 

#merge t1 and t2 return node new node to be conencted by parent
def helper(t1, t2): 
    if not t1 and not t2: return
    if not t1: return t2 
    if not t2: return t1
    
    t1.val = t1.val + t2.val
    t1.left = helper(t1.left, t2.left)
    t1.right = helper(t1.right, t2.right)
    return t1


# Input:
# 	Tree 1                     Tree 2
#           1                         2
#          / \                       / \                            
#         3   2                     1   3                        
#        /                           \   \                      
#       5                             4   7                  
# Output: 
# Merged tree:
# 	     3
# 	    / \
# 	   4   5
# 	  / \   \ 
# 	 5   4   7
# 

# if t1 has no left tree, t2 has left tree, 
# you have to maintain r to connect to t2's left
# if t1 has no right tree , t2 has right tree
# you have to conenct t1.right to t2 right
# if t1 has not children, connect whatever t2 has

# r1.left.val if r1.left else 0 + r2.left.val if r2.left else 0 
#  
def mergeTrees_iter(t1, t2):
    pass
