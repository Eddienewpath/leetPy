import collections
class Node(object):
    def __init__(self, val, left, right, next):
        self.val = val
        self.left = left
        self.right = right
        self.next = next

# space o(n)
def connect(root):
    if not root: return root
    q = collections.deque()
    q.append(root)

    while q:
        size = len(q)
        tmp = runner = Node(-1, None, None, None)
        while size:
            front = q.popleft()
            runner.next = front
            runner = runner.next
            
            if front.left: 
                q.append(front.left)
        
            if front.right:
                q.append(front.right)
            size -= 1
        tmp.next = None
    return root


def connect_recur(root):    
    def helper(root):
        if not root: return
        if root.left:
            root.left.next = root.right
            if root.next:
                root.right.next = root.next.left
        helper(root.left)
        helper(root.right)
    helper(root)
    return root


# O(1）space
def connect_iter(root):
    if not root: return root
    r = root
    while r.left:
        runner = r  
        while runner:
            runner.left.next = runner.right
            if runner.next: 
                runner.right.next = runner.next.left
            runner = runner.next
        r = r.left
    return root 

        
        
        
           


