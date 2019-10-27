import collections
def findDuplicateSubtrees(root):
    counter = collections.Counter()
    res = []
    def helper(root):
        if not root: return '#'
        serial = '{},{},{}'.format(root.val, helper(root.left), helper(root.right))
        counter[serial] += 1
        if counter[serial] == 2:
            res.append(root)
        return serial
    helper(root)
    return res
        



""" Intuition """
""" 
if I do this problem usual way, from bottom up, I need to create a function
separately to compare if two nodes are structrually and numerically identical
and that will create a recurion inside recursion bad space complexity
instead, we can serialize the tree(subtree) into string, and store in hashmap
this will largely reduce the space because you can directly compare two string see if 
they are same. 

if a serial has appaer more than twice, meanign its duplicated and added to result
"""