
class ListNode(object):
    def __init__(self, x):
        self.val = x
        self.next = None


# def addTwoNumbers(l1, l2):
#     h1, h2, carry = l1, l2, 0
#     head = pre = ListNode(-1)
#     while h1 or h2: 
#         total = (h1.val if h1 else 0) + (h2.val if h2 else 0) + carry 
#         carry = total//10
#         pre.next = ListNode(total%10)
#         pre = pre.next
#         if h1:
#             h1 = h1.next  
#         if h2: 
#             h2 = h2.next
#     if carry: 
#         pre.next = ListNode(1)     
#     return head.next
def addTwoNumbers(l1, l2):
    head = pre = ListNode(0) 
    carry = 0
    while l1 or l2 or carry: 
        v1, v2 = 0, 0
        if l1: 
            v1 = l1.val 
            l1 = l1.next
        if l2 : 
            v2 = l2.val 
            l2 = l2.next
        total = v1 + v2 + carry
        carry, val = divmod(total, 10)
        pre.next = ListNode(val)
        pre = pre.next
    return head.next

# n8 = ListNode(8)
# n1 = ListNode(1)
# n8.next = n1
# n0 = ListNode(0)

node1 = ListNode(2)
node2 = ListNode(4)
node3 = ListNode(3)
node1.next = node2
node2.next = node3

n1 = ListNode(5)
n2 = ListNode(6)
n3 = ListNode(4)
n1.next = n2 
n2.next = n3

head = addTwoNumbers(node1, n1)

while head: 
    print(head.val)
    head = head.next
    



# -1->7->0->8

# Input: (2 -> 4 -> 3) + (5 -> 6 -> 4)
# Output: 7 -> 0 -> 8
# Explanation: 342 + 465 = 807.

# two part first add them then reverse linkedlist

