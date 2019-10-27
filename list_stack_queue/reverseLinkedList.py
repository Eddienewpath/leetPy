class Node:
    def __init__(self, val):
       self.val = val
       self.next = None




def reverse(head):
    pre, cur, suc = None, head, None
    while cur: 
        pre = cur
        cur = cur.next
        pre.next = suc
        suc = pre
    return suc


# 1->2->3->4->5
# algo: got the the end and return the new head, then do following
#  4->5->4, then 4 -x-> 5->4
def recur_reverse(head):
    if not head or not head.next: return head
    new_head = recur_reverse(head.next)
    head.next.next = head
    head.next = None
    return new_head
     


first = Node(1)
second = Node(2)
third = Node(3)
fourth = Node(4)
fifth = Node(5)
first.next = second
second.next = third 
third.next = fourth
fourth.next = fifth

new_head = recur_reverse(first)


while new_head: 
    print(new_head.val)
    new_head = new_head.next

# output [5,4,3,2,1]
