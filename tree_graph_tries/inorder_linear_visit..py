class TreeNode: 
    def __init__(self, val):
        self.left = None 
        self.right = None
        self.val = val

# Morris inorder algorithm 

def inorder(root):
    cur = root
    while cur:
        if not cur.left: 
            print(cur.val)
            cur = cur.right
        else: 
            pre = cur.left
            while pre.right and pre.right != cur: 
                pre = pre.right
            
            # when predecessor of cur is found 
            if not pre.right:
                pre.right = cur
                cur = cur.left
            else:
                # restore to original tree  
                print(cur.val)
                pre.right = None
                cur = cur.right


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

inorder(root)


