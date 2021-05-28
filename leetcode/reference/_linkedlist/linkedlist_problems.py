"""
    1) Reverse singly-linked list
    2) Detect cycle in a list
    3) Merge two sorted lists
    4) Remove nth node from the end
    5) Find middle node
    6) Check palindrome
"""

class Node:
    def __init__(self, val):
        self.val = val
        self.next = None

# 1
def reverse_linked_list(head):
    prev, cur = None, head 
    dummy = None
    while cur: 
        prev = cur 
        cur = cur.next
        prev.next = dummy
        dummy = prev
    return dummy

first = Node(1)
second = Node(2)
third = Node(3)
fourth = Node(4)
fifth = Node(5)
sixth = Node(6)
first.next = second 
second.next = third
third.next = fourth
fourth.next = fifth
fifth.next = sixth
# 1-2-3-4-5->3

# d = reverse_linked_list(first)

# while d:
#     print(d.val, end='')
#     d = d.next 
# print()

# 2
def has_cycle(head):
    slow, fast = head, head 
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast: return True 
    return False 

# t = has_cycle(first)
# print(t)

# first.next = third
# third.next = fifth
# second.next = fourth
# fourth.next = sixth

# 3
def merge_sorted_lists(h1, h2):
    p1, p2, p3 = h1, h2, Node(-1)
    d = p3
    while p1 or p2: 
        if p1 and p2: 
            if p1.val > p2.val:
                p3.next = p2
                p2 = p2.next
            else:
                p3.next = p1 
                p1 = p1.next
            p3 = p3.next
        elif p1: 
            p3.next = p1
            break
        else:
            p3.next = p2 
            break
    return d.next

# h = merge_sorted_lists(first, second)
# print(h.val)
# while h:
#     print(h.val, end='-')
#     h = h.next
# print()

# 123456

# 4
def remove_node_from_end(head, n):
    slow, fast, prev = head, head, Node(-1)
    d  = prev
    d.next = head
    while n > 0 and fast:
        fast = fast.next
        n -= 1
    if not fast: return d.next
    while fast:
        prev = slow 
        slow = slow.next 
        fast = fast.next 
    prev.next = slow.next
    return d.next
    

# h = remove_node_from_end(first, 6)
# while h:
#     print(h.val)
#     h = h.next



# 5
def find_middle(head):
    if not head: return head
    slow, fast = head, head
    while fast and fast.next: 
        slow = slow.next 
        fast = fast.next.next
    return slow


# a = find_middle(first)
# print(a.val)


# 6
# 123321
# 12321
def is_palindrome(head):
    slow, fast, prev = head, head, None
    while fast and fast.next:
        prev = slow 
        slow = slow.next 
        fast = fast.next.next
    
    second_half = slow
    if fast:
        second_half = slow.next
    prev.next = None
    
    new_head, cur = None, second_half
    while cur:
        tmp = cur
        cur = cur.next
        tmp.next = new_head
        new_head = tmp
    
    while head: 
        if head.val != new_head.val: return False
        head = head.next 
        new_head = new_head.next

    return True 


n1 = Node(1)
n2 = Node(2)
n3 = Node(3)
# n4 = Node(3)
n5 = Node(2)
n6 = Node(1)
n1.next = n2 
n2.next = n3 
# n3.next = n4 
# n4.next = n5
n3.next = n5 
n5.next = n6 

t = is_palindrome(n1)
print(t)
