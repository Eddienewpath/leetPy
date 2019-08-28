class Solution:
#     def maxSumTwoNoOverlap(self, A, L, M):
#         max_ans = -1
#         for i in range(len(A)):
#             if i+1 < M and len(A)-i-L >= M:
#                 max_ans = max(max_ans, sum(A[i:i+L]) + self.subarray_sum(A[i+L:], M))
#             elif len(A)-i-L < M and i+1 >= M: 
#                 max_ans = max(max_ans, sum(A[i:i+L]) + self.subarray_sum(A[:i], M))
#             else: 
#                 max_ans = max(max_ans, sum(A[i:i+L]) + self.subarray_sum(A[:i], M) + self.subarray_sum(A[i+L:], M))
#         return max_ans



#     def subarray_sum(self, arr, le):
#         s0 = sum(arr[0:le])
#         i, j, max_sum = 0, le, -1
#         while j < len(arr):
#             max_sum = max(s0, s0-arr[i]+arr[j])
#             s0 = s0-arr[i]+arr[j]
#             i += 1
#             j += 1 
#         return max_sum


#     # print(subarray_sum([3,8,1,3,2,1,8,9,0], 3))

# a=[3, 8, 1, 3, 2, 1, 8, 9, 0]
# s = Solution()
# print(s.maxSumTwoNoOverlap(a,3,2))
