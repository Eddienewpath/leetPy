# def deleteNode(node):  
#     runner = node
#     pre = node
#     while runner.next: 
#         runner.val = runner.next.val
#         pre = runner
#         runner = runner.next
#     pre.next = None   

# better way coz you dont need to swap all the nodes coz latter nodes are in correct chain
# only the node and the next node effected when performing this deletion.
def deleteNode(node):
    node.val = node.next.val
    node.next = node.next.next

