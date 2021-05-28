def isSubtree(self, s, t):
        if not s: return False
        if self.is_same(s, t):
            return True
        return self.isSubtree(s.left, t) or self.isSubtree(s.right, t)


def is_same(self, r1, r2):
    if not r1 and not r2: return True 
    if not r1 or not r2: return False 
    return r1.val == r2.val and self.is_same(r1.left, r2.left) and self.is_same(r1.right, r2.right)
    




def levelOrder(self, root):
    res = []
    if not root: return res
    self.level_dfs(root, 0, res)
    return res
        
# dfs way of level order traversal is actually a preorder traversal with addtional condition to create the list and adding the same level result to that list
def level_dfs(self, root, level, res):
    if not root: return
    # when level == to size of res meaning the lists in the res are not enough coz on the next line we need to use the level to find the right list to append current node
    # or we at at new level, so we need a new list to hold that level's items.
    if level == len(res):
        res.append([])
    """ preorder traversal """
    res[level].append(root.val)
    self.level_dfs(root.left, level+1, res)
    self.level_dfs(root.right, level+1, res)
    
    

# 107 reverse level order 
def levelOrderBottom(self, root):
        res = []
        if not root: return res
        self.dfs(root, 0, res)
        return res
    

def dfs(self, root, level, res):
    if not root:return
    if level == len(res):
        res.insert(0, [])
        
    res[len(res)-level-1].append(root.val)
    self.dfs(root.left, level+1, res)
    self.dfs(root.right, level+1, res)




def hasPathSum(self, root, sum):
    if not root: return False
    if not root.left and not root.right and (sum - root.val == 0):
        return True
    return self.hasPathSum(root.left, sum-root.val) or self.hasPathSum(root.right, sum-root.val)
        

# 257 
def binaryTreePaths(self, root):
        res = []
        self.dfs(root, '', res)
        return res

def dfs(self, root, path, res):
    if not root: return 
    if not root.left and not root.right:
        res.append(path+str(root.val))
        return 
    
    self.dfs(root.left, path+str(root.val) + '->', res)
    self.dfs(root.right, path+str(root.val) + '->', res)



# 113
def pathSum(self, root, sum):
    res = []
    self.helper(root, [], res , sum)
    return res 
    
def helper(self, root, path, res, total):
    if not root: return
    if not root.left and not root.right and (total - root.val == 0):
        path.append(root.val)
        res.append(path)
        return
    
    self.helper(root.left, path + [root.val], res, total - root.val)
    self.helper(root.right, path + [root.val], res, total - root.val)
