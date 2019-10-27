# iterative 
# def reverseList(head):
#     runner, pre = head, head
#     front = None
#     while runner:
#         runner = runner.next
#         pre.next = front
#         front = pre
#         pre = runner
#     return front

# recursive my version 
# def reverseList(head):
#     dummy = ListNode(-1)
#     def helper(h):
#         if not h:
#             return dummy    
#         d = helper(h.next)
#         d.next = h
#         if h == head:
#             h.next = None
#             return
#         return h
#     helper(head)
#     return dummy.next

# recursive good version
def reverseList(head):
    if not head or not head.next: 
        return head
    
    front = reverseList(head.next)
    prev = head.next
    prev.next = head
    head.next = None
    return front
