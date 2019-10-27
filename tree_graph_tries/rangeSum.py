def rangeSumBST(root, L, R):
    if not root: return 0
    return rangeSumBST(root.left, L, R) + rangeSumBST(root.right, L, R) + (root.val if  L <= root.val <= R else 0)



# find the total range sum of left and right and decide if root is in range
# ranageSumBST(root.left, L, R) + rangeSumBST(root.right, L, R) + r.val of L<= r.val <= R

# faster with bst branching
def rangeSum(root, L , R):
    if not root:return 0
    if root.val < L: return rangeSum(root.right, L, R)
    if root.val > R : return rangeSum(root.left, L, R)
    return root.val + rangeSum(root.left, L, R) + rangeSum(root.right, L, R)
