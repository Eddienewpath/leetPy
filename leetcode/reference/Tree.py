# Fundamental tree operations. 

class Node: 
    def __init__(self, val):
        self.val = val
        self.left = self.right = None


class Binary_Tree:
    # pre-order traversal
    def preorder_recursive(self, r):
        if not r: return 
        print(r.val)
        self.preorder_recursive(r.left)
        self.preorder_recursive(r.right) 


    def preorder_stack_one(self, r):
        stack = [r]
        while stack:
            top = stack.pop()
            print(top.val)
            if top.right:
                stack.append(top.right)
            
            if top.left:
                stack.append(top.left)



    def preorder_stack_two(self, r):
        stack = []
        while stack or r:
            while r:
                print(r.val)
                stack.append(r)
                r = r.left
            
            if stack: 
                r = stack.pop().right

    # in order traversal
    def inorder_recursive(self, r):
        if not r: return 
        self.inorder_recursive(r.left)
        print(r.val)
        self.inorder_recursive(r.right)   
    

    def inorder_stack(self, r):
        stack = []
        while stack or r:
            while r:
                stack.append(r)
                r = r.left
            
            if stack: 
                r = stack.pop()
                print(r.val)
                r = r.right
    

    # post order traversal
    def postorder_recursive(self, r):
        if not r: return 
        self.postorder_recursive(r.left)
        self.postorder_recursive(r.right) 
        print(r.val)  
    
    
    # reverse rotated(process right first) in order traversal is the iterative of post order traversal
    def postorder_stack(self, r):
        import collections
        res = []
        stack = [r]
        while stack:
            top = stack.pop()
            res.append(top.val)
            if top.left:
                stack.append(top.left)
            if top.right:
                stack.append(top.right)

        while res:
            print(res.pop())

    
    def levelorder(self, r):
        res = []
        self.level_recur(r, 0, res)
        print(res)

        res = []
        self.level_iter(r, res)
        print(res)
    

    def level_recur(self, r, level, res):
        if not r: return

        if len(res)-1 < level:
            res.append([])
        res[level].append(r.val)

        self.level_recur(r.left, level+1, res)
        self.level_recur(r.right, level+1, res)



    def level_iter(self, r, res):
        if not r: return 
        import collections
        queue = collections.deque([r])
        while queue:
            size = len(queue)
            tmp = []
            for _ in range(size):
                front = queue.popleft()
                tmp.append(front.val)
                if front.left:
                    queue.append(front.left)
                if front.right:
                    queue.append(front.right)
            res.append(tmp)



    # for normal binary tree
    def lowest_common_ancestor(self, r, p, q):
        if not r: return
        if r.val == p.val or r.val == q.val:
            return r 

        left = self.lowest_common_ancestor(r.left, p, q)
        right = self.lowest_common_ancestor(r.right, p, q)

        return r if (left and right) else (left or right)


    # inorder traversal iterative with stack + build child to parent mapping. 
    # add all p's ancestors into a set, and start from q and go upward on its ancestor and the first ancestor of q in p's ancestor is the LCA
    def lowest_common_ancestor_iter(self, r, p, q):
        parent = {r: None}
        stack = [r]
        while p not in parent or q not in parent:
            top = stack.pop()
            if top.left:
                stack.append(top.left)
                parent[top.left] = top
            if top.right:
                stack.append(top.right)
                parent[top.right] = top
        p_ancestors = set()
        while p:
            p_ancestors.add(p)
            p = parent[p]
        
        while q not in p_ancestors:
            q = parent[q]
        return q



    # check if a binary tree is bst
    # trick here:  r.left.val < r.left.right.val < r.val 
    def isValidBST(self, root):
        return self.is_bst(root, None, None)
    
    
    def is_bst(self, r, left_max, right_min):
        if not r: 
            return True
        
        if (left_max != None and r.val >= left_max) or (right_min != None and r.val <= right_min):
            return False
        
        return self.is_bst(r.left, r.val, right_min) and self.is_bst(r.right, left_max, r.val)
        




# root = Node(1)
# root.left = Node(2)
# root.right = Node(3)
# root.left.left = Node(4)
# root.left.right = Node(5)

# t = Binary_Tree()
# # t.preorder_stack_two(root)
# # t.preorder_recursive(root)
# t.inorder_stack(root) # 4, 2, 5, 1, 3
# t.postorder_stack(root) # 4, 5, 2, 3, 1

# t.levelorder(root)

# left < root < right
class BST(Tree):
    def __init__(self):
        self.root = None

    # search for target
    def search(self, target):
        return self.search_iter(self.root, target)


    def search_recursive(self, root, target):
        if not root: return 
        if root.val == target: return root

        if root.val < target:
            return self.search_recursive(root.right, target)
        else:
            return self.search_recursive(root.left, target)

    
    def search_iter(self, root, target):
        while root and target != root.val:
            if root.val < target:
                root = root.right
            else:
                root = root.left
        return root != None

# next greater.
    def find_successor(self, node):
        mapping = {self.root: None} 
        self.kid_parent_mapping(self.root, mapping)
        if node.right:
            cur = node.right
            while cur.left:
                cur = cur.left
            return cur
        parent = mapping[node]
        #if node is right child of parent
        while parent and parent.right == node:
            node = parent
            parent = mapping[parent]
        return parent
        


    def kid_parent_mapping(self, root, mapping):
        if not root: return
        if root.left:
            mapping[root.left] = root
        if root.right:
            mapping[root.right] = root
        self.kid_parent_mapping(root.left, mapping)
        self.kid_parent_mapping(root.right, mapping)




    # insert a node with value of val, return the root of the tree. 
    def insert(self, target):
        node = Node(target)
        self.root = self.insert_recursive(self.root, node)
        return self.root
# insert node into the bst with root, return the updated root
# base case: insert into null tree, return the updated root which is the node.
# essentially, inertion is searching for a null tree to put the target node and do not break the existing structure. 
    def insert_recursive(self, root, node):
        if not root: return node
        if root.val < node.val:
            root.right = self.insert_recursive(root.right, node)
        if root.val > node.val:
            root.left = self.insert_recursive(root.left, node)
        return root



    # delete the node with target val 
    def delete(self, target):
        node = Node(target)
        return self.del_recur(self.root, node)
    

   # delete the target and return the updated root
   # after deletion, the tree still has to be bst.
    def del_recur(self, root, node):
        if not root: return
        if root.val == node.val:
            if root.left and root.right:
                self.placeleftmost(root.right, root.left)
                return root.right 
            else:
                return root.left or root.right
    
        if root.val < node.val:
            root.right = self.del_recur(root.right, node)
        else:
            root.left = self.del_recur(root.left, node)
        return root



    def placeleftmost(self, root, node):
        while root.left:
            root = root.left
        root.left = node 

    
    def printNodes(self):
        self.inorder_recursive(self.root)


    def inorder_recursive(self, r):
        if not r: return 
        self.inorder_recursive(r.left)
        print(r.val)
        self.inorder_recursive(r.right)   




#      4
#     / \
#    2   5
#   / \   \    
#  1   3   6
# /
#0

# bst = BST()
# for i in [4, 2, 5, 1, 3, 6]:
#     bst.insert(i)

# print(bst.find_successor(bst.root.left.left).val)





class TrieNode:
    def __init__(self, c=''):
        self.char = c
        self.children = {} # char -> node
        self.flag = False

# or prefix tree
class Trie:
    def __init__(self):
        self.root = TrieNode()

# insert word into the trie
    def insert(self, word):
        r = self.root
        for c in word:
            if c not in r.children:
                r.children[c] = TrieNode(c)
            r = r.children[c]
        r.flag = True


# check given string is word or not
    def search(self, st):
        r = self.root
        for c in st:
            if c not in r.children: 
                return False
            r = r.children[c]
        return r.flag 



# return boolean, check whether a word starts with prefix
    def startswith(self, prefix):
        r = self.root
        for c in prefix:
            if c not in r.children:
                return False 
            r = r.children[c]
        return True



# using array/list to replace dictionary
# when the problem is about string and involve hashtable, usually you can use array to optimize the solution instead of use a full blown hashmap/dictionary
# because its only 26 or 128 characters.
class TrieNode:
    def __init__(self, c=''):
        self.char = c
        self.children = [None] * 26
        self.flag = False


class Trie:
    def __init__(self):
        self.root = TrieNode()

    # insert word into the trie
    def insert(self, word):
        r = self.root
        for c in word:
            idx = ord(c) - ord('a')
            if not r.children[idx]:
                r.children[idx] = TrieNode(c)
            r = r.children[idx]
        r.flag = True

    # check given string is word or not
    def search(self, st):
        r = self.root
        for c in st:
            idx = ord(c) - ord('a')
            if not r.children[idx]: 
                return False
            r = r.children[idx]
        return r.flag 

    # return boolean, check whether a word starts with prefix
    def startswith(self, prefix):
        r = self.root
        for c in prefix:
            idx = ord(c) - ord('a')
            if not r.children[idx]: 
                return False 
            r = r.children[idx]
        return True








