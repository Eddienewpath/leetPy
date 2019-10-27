class Node(object):
    def __init__(self, val, next):
        self.val = val
        self.next = next


# two cases: 
# when n1.val <= insertVal <= n.val
# when insertVal is max or min 
def insert(head, insertVal):
    if not head: return Node(insertVal, head)
    node = Node(insertVal, None)
    pre, cur = head, head.next
    while 1: 
        if pre.val <= insertVal <= cur.val: 
            break
        if pre.val > cur.val and (insertVal <= cur.val or insertVal > pre.val):
            break
        pre = pre.next
        cur = cur.next
        if pre == head: 
            break
    pre.next = node
    node.next = cur
    return head









#     if not head:
#         head = Node(insertVal, head)
#         return head

#     h = Node(insertVal, None)
#     if head.next == head:   
#         if head.val > insertVal: 
#             h.next = head
#             head.next = h
#             return head
#         else:
#             head.next = h
#             h.next = head
#             return head
            
    
# #         find the end of the list, where val is max
#     runner = head
#     while runner.next.val >= runner.val: 
#         runner = runner.next
        
# #       original head found here, min 
#     ori = runner.next
#     if ori.val > insertVal: 
#         runner.next = h
#         h.next = ori
        
# #       else if ori is still the min
# #       find the first node that is greater than given numeber and insert into
#     runner = ori
#     pre = ori 
#     while runner.val <= insertVal: 
#         pre = runner
#         runner = runner.next

#     pre.next = h
#     h.next = runner
    
#     return head
    


head = Node(3, None)
head.next = Node(4, None)
head.next.next = Node(1,None)
head.next.next.next = head


ahead = insert(head, 2)
# print(ahead.val)

runner = ahead.next
pre = head
while runner != ahead:
    print(runner.val)
    pre = runner
    # print(pre.val)
    runner = runner.next
print(runner.val)
