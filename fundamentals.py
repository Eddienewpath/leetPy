import heapq
import collections
from collections import deque

# space k, space 0 to fold all codes 

""" K sum problems """
# Given nums = [2, 7, 11, 15], target = 9,
# Because nums[0] + nums[1] = 2 + 7 = 9,
# return [0, 1].
def twoSum(nums, target):
    m = {} # [k=comp, v=index] store complememt of later elements 
    for i, v in enumerate(nums):
        if v not in m: 
            m[target - v] = i
        return [m[v], i]


# Input: numbers = [2, 7, 11, 15], target = 9
# Output: [1, 2]
# Explanation: The sum of 2 and 7 is 9. Therefore index1 = 1, index2 = 2.
# Your returned answers(both index1 and index2) are not zero-based.
# You may assume that each input would have exactly one solution and you may not use the same element twice.

""" conclusion: when we see sorted array, we need to think of two pointers and binary search
 """

def twoSum_sorted_binary_search(numbers, target):
    for i in range(len(numbers)):
        comp = target - numbers[i]
        start, end = i+1, len(numbers)-1
        while start <= end:
            mid = (start + end)//2
            if numbers[mid] == comp:
                return [i+1, mid+1]
            elif numbers[mid] < comp:
                start = mid + 1
            else: 
                end = mid - 1
    return []


def twoSum_sorted_two_pointers(numbers, target):
    left, right = 0, len(numbers)-1
    while left < right:
        tmp = numbers[left] + numbers[right]
        if tmp == target: return [left+1, right+1]
        elif tmp < target: left += 1
        else: right -= 1
    return []


# The solution set must not contain duplicate triplets.
# Given array nums = [-1, 0, 1, 2, -1, -4],
# A solution set is:
# [
#     [-1, 0, 1],
#     [-1, -1, 2]
# ]


def threeSum(nums):
    res = []
    nums.sort() # so that you can use two pointer 
    for i in range(len(nums)):
        #if there are dup, use the outer one ignore the inner ones 
        if i > 0 and nums[i] == nums[i-1]: continue
        target = 0 - nums[i]
        left, right = i+1, len(nums)-1 # two pointers
        while left < right: 
            if nums[left] + nums[right] == target:
                # we need to skip the dup, keep the most inner ones that sum to target
                while left < right and nums[left] == nums[left+1]: left += 1
                while left < right and nums[right] == nums[right-1]: right -= 1
                # append result
                res.append([nums[i], nums[left], nums[right]])
                # update both left and right to continue searching
                left += 1
                right -= 1
            elif nums[left] + nums[right] < target:
                left += 1
            else: right -= 1
    return res 

# print(threeSum([-1,0,1,2,-1,-4]))

def threeSumClosest(nums, target):
    closest = float('inf')
    ans = 0
    nums.sort()
    for i in range(len(nums)):
        total = 0
        left, right = i+1, len(nums)-1
        while left < right:
            total = nums[i] + nums[left] + nums[right]
            # narrow the possibility as many as possible
            if total < target:
                left += 1
            elif total > target: 
                right -= 1
            else: 
                # smallest difference is 0, just return
                return target
 
            diff = abs(target - total)
            if diff < closest:
                closest = diff
                ans = total
    return ans 

# print(threeSumClosest([0,1,2], 3))

def fourSum(nums, target):
    # recursively reduce K sum to 2 sum, insert result
    def kSum(nums, target, k, tmp, results):
        if len(nums) < k or k < 2:
            return
        if k == 2:
            # two pointer
            left, right = 0, len(nums)-1
            while left < right:
                if nums[left] + nums[right] == target:
                    while left < right and nums[left] == nums[left+1]:
                        left += 1
                    while left < right and nums[right] == nums[right-1]:
                        right -= 1
                    results.append(tmp + [nums[left], nums[right]])
                    left += 1
                    right -= 1
                elif nums[left] + nums[right] < target:
                    left += 1
                else:
                    right -= 1
        else:
            for i in range(len(nums)-k+1):
                if i > 0 and nums[i] == nums[i-1]:
                    continue
                kSum(nums[i+1:], target-nums[i], k-1, tmp + [nums[i]], results)

    nums.sort()
    results = []
    kSum(nums, target, 4, [], results)
    return results


""" Sort """
"""
worst case: when the array is already sorted and we always picked greatest or smallest element as pivot
the run time is O(n^2). but in reality this rarely happend
best case: is when we always pick the middle element as the pivot, that will be O(nlgn)
"""
# put pivot element at its right place
# and do the same to the elements of left and right side of pivot elements 
def quick_sort(arr):
    # put all the elements less than pivot to the left of pivot
    def partition(arr, i, j): 
        par = i
        # for this implementation we user last position as pivot index 
        pivot = arr[j]
        for k in range(i, j+1): 
            if arr[k] <= pivot: 
                arr[k], arr[par] = arr[par], arr[k]
                par += 1
        return par-1 

    def quick_sort_helper(arr, start, end): 
        if start > end: return 
        p = partition(arr, start, end)
        # recursively do it to subarrays
        quick_sort_helper(arr, start, p-1)
        quick_sort_helper(arr, p+1, end)

    quick_sort_helper(arr, 0, len(arr)-1)
    return arr
# print(quick_sort([6,4,5,3,2,1,2,3,4]))

"""
merge sort is generally consider the best when the data is huge and store in external storages  
theta(nlgn) 
"""
# 3, 2, 4, 1, 5
def merge_sort(arr):
    # one element list is sorted so return 
    if len(arr) <= 1: return [arr[0]]
    half = len(arr)//2
    left = merge_sort(arr[:half])
    right = merge_sort(arr[half:])
    return merge(left, right)

# merge two sorted array and return 
def merge(a1, a2):
    res = [None]*(len(a1) + len(a2))
    i = j = k = 0
    n = len(a1)
    m = len(a2)
    while i < n or j < m:
        if i >= n and j < m: 
            while j < m: 
                res[k] = a2[j]
                j += 1
                k += 1
        elif i < n and j >= m: 
            while i < n: 
                res[k] = a1[i]
                i += 1
                k += 1
        else: 
            if a1[i] <= a2[j]: 
                res[k] = a1[i]
                i += 1
            else: 
                res[k] = a2[j]
                j += 1
            k += 1

    return res        
# print(merge_sort([5,4,3,2,1,0]))

""" 
heap sort
"""
class MaxHeap(object):
    def __init__(self, N, arr): 
        self.arr = arr
        self.size = N 
        self.build()

    # percolate up 
    def build(self):
        for i in range(self.size//2)[::-1]: 
            self.heapify(i)
    
    # float down
    def heapify(self, i):
        if i > self.size: return
        left = 2*i+1
        right = 2*i+2
        largest = -1 
        if left < self.size and self.arr[i] < self.arr[left]: 
            largest = left
        else: 
            largest = i
        
        if right < self.size and self.arr[right] > self.arr[largest]: 
            largest = right
        
        if largest != i: 
            self.arr[i], self.arr[largest] = self.arr[largest], self.arr[i]
            self.heapify(largest)



    def heap_sort(self):
        n = self.size
        for i in range(n)[::-1]: 
            self.arr[0], self.arr[i] = self.arr[i], self.arr[0]
            self.size -= 1
            self.heapify(0)
        

# arr = [3,2,4,1,5,0]
# mh = MaxHeap(len(arr), arr)
# mh.heap_sort()
# print(arr)
     

""" 
cons: O(n^2)
pros: only make O(n) swap, it is good for memory writes are costly operation and only user constant space
"""
def selection_sort(arr):
    for i in range(len(arr)):
        min_idx = i
        min_val = arr[min_idx]
        for j in range(i, len(arr)):
            if arr[j] < min_val: 
                min_idx = j 
                min_val = arr[j]
        arr[i], arr[min_idx] = arr[min_idx], arr[i]
    return arr

# print(selection_sort([3,2,4,1,5]))  


"""
pros: It is in place and easy to detect small errs by swap two elements at a time
cons: normal bubble sort will have worse and average time o(n^2) even when the input is sorted
"""
# a bit optimized. when array is sorted, now takes O(n)
def bubble_sort(arr):
    for i in range(len(arr)):
        swapped = False
        for j in range(len(arr)-i):
            if j > 0 and arr[j] < arr[j-1]:
                arr[j], arr[j-1] = arr[j-1], arr[j]
                swapped = True
        if not swapped: break
    return arr 
                

# print(bubble_sort([3,2,4,1,5]))
"""
pros: when input is small, it's quite efficient and reduced swaps and stable
cons: when input is larget, its slow 
"""
def insertion_sort(arr):
    for i in range(1, len(arr)):
        j = i
        while j > 0 and arr[j] < arr[j-1]: 
            arr[j], arr[j-1] = arr[j-1], arr[j]
            j -= 1
    return arr

# print(insertion_sort([12, 11, 13, 5, 6]))

""" Array 2da-array"""
# Input:
# [[1, 2, 3],
#  [4, 5, 6],
#  [7, 8, 9]]
def spiralOrder(matrix):
    res = []
    n = len(matrix)
    m = len(matrix[0])
    if not matrix: return res 
    up = 0 
    down = n-1
    left = 0
    right = m-1
    
    while left <= right and up <= down: 
        # left - > right 
        for i in range(left, right+1):
            res.append(matrix[up][i])
        up += 1
        # top -> down
        for i in range(up, down+1):
            res.append(matrix[i][right])
        right -= 1

        # right -> left 
        if up <= down: 
            for i in range(right, left-1, -1): 
                res.append(matrix[down][i])
            down -= 1 

        # down -> top
        if left <= right:
            for i in range(down, up-1, -1): 
                res.append(matrix[i][left])
            left += 1 
    return res

# a = [[1, 2, 3, 4],
#     [5, 6, 7, 8],
#     [9, 10, 11, 12]]
# # Output: [1, 2, 3, 6, 9, 8, 7, 4, 5]
# print(spiralOrder(a))
# basically the same as above problem
def generateMatrix(n):
    up = 0
    down = n-1
    left = 0
    right = n-1
    k = 1
    matrix = [[0 for i in range(n)] for j in range(n)]

    while left <= right and up <= down:
        # left - > right
        for i in range(left, right+1):
            matrix[up][i] = k
            k += 1
        up += 1
        # top -> down
        for i in range(up, down+1):
            matrix[i][right] = k
            k += 1
        right -= 1

        # right -> left
        if up <= down:
            for i in range(right, left-1, -1):
                matrix[down][i] = k
                k += 1
            down -= 1

        # down -> top
        if left <= right:
            for i in range(down, up-1, -1):
                matrix[i][left] = k
                k += 1
            left += 1
    return matrix

# print(generateMatrix(0))

""" Backtracking """
def permute(nums):
    res = []
    def backtrack(nums, path): 
        if not nums: res.append(path)
        for i in range(len(nums)):
            # building a path starts with [i], and backtracking the rest of elements
            backtrack(nums[:i]+nums[i+1:], path+[nums[i]])
    backtrack(nums, [])
    return res


def permuteUnique(nums):
    res = []
    def backtrack(nums, path): 
        if not nums: res.append(path)
        for i in range(len(nums)):
            if i+1 < len(nums) and nums[i] == nums[i+1]: continue
            backtrack(nums[:i]+nums[i+1:], path+[nums[i]])
    nums.sort()
    backtrack(nums, [])
    return res 


# bottom -> up
def subsets_bottom_up(nums):
    if not nums: return [[]] 
    for i in range(len(nums)):
        res = subsets_bottom_up(nums[i+1:])
        res += [[nums[i]] + l for l in res]
        return res

# dfs optimized 
def subsets_top_down(nums):
    def build_path(nums, start, path, res): 
        res.append(path)
        for i in range(start, len(nums)):
            build_path(nums, i+1, path+[nums[i]], res)
    res = []
    build_path(nums, 0, [], res)
    return res
         

def subsetsWithDup(nums):
    def build_path_skip_dup(nums, start, path, res):
        res.append(path)
        for i in range(start, len(nums)):
            if i-1 >= start and nums[i] == nums[i-1]:continue # skip the later dup 
            build_path_skip_dup(nums, i+1, path+[nums[i]], res)
    nums.sort()
    res = []
    build_path_skip_dup(nums, 0, [], res)
    return res


def combinationSum(candidates, target):
    def dfs(candidates, target, path, start, res):
        if target < 0 : return 
        if target == 0: 
            res.append(path)
            return
        for i in range(start, len(candidates)):
            dfs(candidates, target-candidates[i], path+[candidates[i]], i, res) 
    res = []
    dfs(candidates, target, [], 0, res)
    return res


def combinationSum2(candidates, target):
    def dfs(candidates, target, path, start, res):
        if target < 0: return
        if target == 0:
            res.append(path)
            return
        for i in range(start, len(candidates)):
            if i-1 >= start and candidates[i] == candidates[i-1]: continue
            dfs(candidates, target -
                candidates[i], path+[candidates[i]], i+1, res)
    candidates.sort()
    res = []
    dfs(candidates, target, [], 0, res)
    return res
              

def combinationSum3(k, n):
    def dfs(k, n, arr, path, res, start):
        if k < 0 or n < 0: return
        if k == 0 and n == 0:
            res.append(path)
            return
        for i in range(start, len(arr)):
            dfs(k-1, n-arr[i], arr, path+[arr[i]], res, i+1)

    arr = [1,2,3,4,5,6,7,8,9]
    res = []
    dfs(k, n, arr, [], res, 0)
    return res 
        

def generateParenthesis(n):
    pass


def generateAbbreviations(word):
    pass


""" Two Pointers, sliding window, substring"""
# p-26
def removeDuplicates(nums):
    avail = 1
    for i in range(len(nums)):
        if i > 0 and nums[i] != nums[i-1]:
            nums[avail] = nums[i]
            avail += 1
    return avail


def findAnagrams(s, p):
    dic_p, dic_s = {}, {}
    for c in p:
        dic_p[c] = dic_p.get(c, 0)+1
    start, end = 0, 0 
    n = len(s)
    res = []
    while end < n:
        dic_s[s[end]] = dic_s.get(s[end], 0)+1
        if dic_p == dic_s:
            res.append(start)
        end += 1
        if end - start + 1 > len(p):
            dic_s[s[start]] -= 1
            if dic_s[s[start]] == 0:
                del dic_s[s[start]]
            start += 1
    return res 

# same code as find all anagram
def checkInclusion(s1, s2):
    dic1, dic2 = {}, {}
    n = len(s2)
    m = len(s1)
    start, end = 0, 0 
    for c in s1:
        dic1[c] = dic1.get(c, 0) + 1
    
    while end < n: 
        dic2[s2[end]] = dic2.get(s2[end], 0) + 1
        if dic1 == dic2: return True
        end += 1
        if end - start + 1 > m:
            dic2[s2[start]] -= 1
            if dic2[s2[start]] == 0: 
                del dic2[s2[start]]
            start += 1 
    return False 


def characterReplacement(s, k):
    count = {}
    max_count, max_len = 0, 0
    start, res = 0, 0
    for end in range(len(s)):
        count[s[end]] = count.get(s[end], 0)+1
        max_count = max(max_count, count[s[end]])
        while end - start + 1 - max_count > k:
            count[s[start]] -= 1
            start += 1
        max_len = max(max_len, end - start + 1)
    return max_len


# greedy tricky
def partitionLabels(S):
    # last occurence of each character
    dic = {c:i for i, c in enumerate(S)} # fast way to create dictionary
    j = anchor = 0
    res = []
    for i, c in enumerate(S):
        j = max(j, dic[c])

        if i == j: 
            res.append(j - anchor + 1)
            anchor = i + 1 
    return res 


# do it again 
def numSubarrayProductLessThanK(nums, k):
    if k <= 1: return 0
    prod, count, i = 1, 0, 0
    n = len(nums)
    for j in range(n):
        prod *= nums[j]
        while prod >= k:
            prod /= nums[i]
            i += 1
        count += j - i + 1
    return count


def minSubArrayLen(s, nums):
    minLen = float('inf') 
    i, j = 0, 0
    total, n = 0, len(nums)
    while i < n or j < n: 
        if j < n and total < s: 
            total += nums[j]
            j += 1
        else:
            if total >= s and j - i < minLen:
                minLen = j - i
            total -= nums[i]
            i += 1
    return minLen if minLen != float('inf') else 0


def minWindow(s, t):
    if not s or not t: return ''
    dic = {}
    for c in t:
        dic[c] = dic.get(c, 0)+1 
    start, end = 0, 0
    n, m = len(s), len(t)
    min_size , res = n, ''
    while end < n: 
        if s[end] in dic: 
            if dic[s[end]] > 0: m -= 1 
            dic[s[end]] -= 1 # detect dup if < 0

        end += 1
        while m == 0: 
            if min_size >= end - start:
                min_size = end - start
                res = s[start : end]
                
            if s[start] in dic:
                dic[s[start]] += 1 
                # if prev is 0, meaning cur char has no dup, increment size by 1 
                if dic[s[start]] > 0: 
                    m += 1 
            start += 1 
    return res


# monotonic queue

def maxSlidingWindow(nums, k):
    dq = deque()
    res = []
    for i in range(len(nums)):
        # see if current max in dq is inside the window
        if dq and dq[0] == i - k: 
            dq.popleft()
        # maintain decreasing stack
        while dq and nums[dq[-1]] < nums[i]: 
            dq.pop()
        dq.append(i)
        if i >= k - 1:
            res.append(nums[dq[0]])
    return res 
            
        
# edge case: "tmmzuxt" handled by start <= dic[c]
def lengthOfLongestSubstring(s):
    dic = {}
    start = 0
    max_len = 0
    for end, c in enumerate(s):
        if c in dic and start <= dic[c]:
            start = dic[c] + 1
        else:
            max_len = max(max_len, end - start + 1)
        dic[c] = end
    return max_len


def lengthOfLongestSubstringTwoDistinct(s):
    max_len = 0
    dic = {}
    start, k = 0, 2
    for end in range(len(s)):
        while k == 0 and s[end] not in dic:
            dic[s[start]] -= 1
            if dic[s[start]] == 0:
                del dic[s[start]]
                k += 1
            start += 1
        if s[end] in dic:
            dic[s[end]] += 1
        else:
            dic[s[end]] = 1
            k -= 1
        max_len = max(max_len, end - start + 1)
    return max_len

# divide and conquor. two pointer solution is trivial
def longestSubstring(s, k):
    if len(s) < k: return 0 
    c = min(set(s), key=s.count)
    if s.count(c) >= k: return len(s)
    return max(longestSubstring(t, k) for t in s.split(c))


def characterReplacement(s, k):
    char_count, max_char, start, ans = {}, 0, 0, 0
    for end in range(len(s)):
        char_count[s[end]] = char_count.get(s[end], 0)+1
        max_char = max(max_char, char_count[s[end]])
        while end - start + 1 - max_char > k:
            char_count[s[start]] -= 1
            start += 1 
        ans = max(ans, end - start + 1)
    return ans


""" Linked List """
def reverseBetween(head, m, n):
    dummy = ListNode(-1)
    dummy.next = head
    runner = dummy
    pre, last, next = None, None, None
    counter = 0
    while runner:
        pre = runner
        runner = runner.next
        counter += 1
        if counter == m:
            last = runner
            while counter <= n:
                tmp = runner
                runner = runner.next
                tmp.next = next
                next = tmp
                counter += 1
            pre.next = next
            last.next = runner
    return dummy.next


def hasCycle(head):
    if not head: return False
    slow = head
    fast = head
    # fast has to be exist then check if it has next
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            return True
    return False

# cyle detection math 
def detectCycle(head):
    if not head: return None
    slow = fast = begin = head
    # fast has to be exist then check if it has next
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            while slow != begin:
                slow = slow.next
                begin = begin.next
            return begin
    return None


def isPalindrome(head):
    if not head: return True

    def compare(h1, h2):
        while h1 and h1:
            if h1.val != h2.val:
                return False
            h1 = h1.next 
            h2 = h2.next
        return True

    def reverse(head):
        next = None
        while head:
            tmp = head
            head = head.next
            tmp.next = next
            next = tmp
        return next

    pre, slow, fast = head, head, head
    while fast and fast.next:
        pre = slow
        slow = slow.next
        fast = fast.next.next
    pre.next = None
    slow = slow.next if fast else slow
    head2 = reverse(slow)
    runner = head
    return compare(head2, runner)


def plusOne_dic(head):
    if not head: return 0
    dummy = ListNode(-1)
    dummy.next = head
    dic = {head: dummy}
    pre = None
    while head:
        pre = head
        head = head.next
        dic[head] = pre

    carry = 1
    while pre.val > 0:
        total = pre.val + carry
        carry = total // 10
        pre.val = total % 10
        pre = dic[pre]

    pre.val = 1 if carry else pre.val
    return dummy if dummy.val == 1 else dummy.next

# carry only affect the digit right before 9, using two pointers to locate the previous not 9 digit before 9 
def plusOne_twoPointers(head):
    if not head: return 0
    dummy = ListNode(0)
    dummy.next = head
    i = j = dummy
    while j.next:
        j = j.next 
        if j.val != 9:
            i = j
    if j.val != 9: 
        j.val += 1 
    else:
        i.val += 1
        i = i.next
        while i: 
            i.val = 0
            i = i.next
    if dummy.val == 0:
        return dummy.next
    return dummy


def addTwoNumbers(l1, l2):
    stack1, stack2 = [], []

    while l1:
        stack1.append(l1.val)
        l1 = l1.next

    while l2:
        stack2.append(l2.val)
        l2 = l2.next

    carry, next = 0, None

    while stack1 or stack2 or carry:
        total = 0
        if stack1:
            total += stack1.pop()
        if stack2:
            total += stack2.pop()
        total += carry
        n = ListNode(total % 10)
        n.next = next
        next = n
        carry = total // 10
    return next


def mergeTwoLists_iter(l1, l2):
    dummy = ListNode(-1)
    runner = dummy
    while l1 and l2:
        if l1.val < l2.val:
            runner.next = l1
            l1 = l1.next
        else:
            runner.next = l2
            l2 = l2.next
        runner = runner.next
        runner.next = None
    runner.next = l1 or l2
    return dummy.next


def mergeTwoLists_recur(l1, l2):
    if not l1 or not l2: return l1 or l2
    if l1.val < l2.val:
        l1.next = mergeTwoLists_recur(l1.next, l2)
        return l1
    else:
        l2.next = mergeTwoLists_recur(l1, l2.next)
        return l2


def oddEvenList(head):
    if not head or not head.next: return head
    even = head
    odd = even.next
    oddhead = odd
    while odd and odd.next:
        even.next = odd.next
        even = even.next
        odd.next = even.next
        odd = odd.next
    even.next = oddhead
    return head

def partition(head, x):
    if not head: return head
    d1, d2 = ListNode(-1), ListNode(-1)
    r1, r2 = d1, d2
    while head:
        if head.val < x:
            r1.next = head
            r1 = r1.next
        else:
            r2.next = head
            r2 = r2.next
        head = head.next
    r2.next = None  # r2.next could point to some other nodes cause cycle
    r1.next = d2.next
    return d1.next

#       intersect at a node or at null
#       should include the null node to end the while condition
def getIntersectionNode(headA, headB):
        h1 = headA
        h2 = headB
        while h1 != h2:
            if not h1:
                h1 = headB
            else:
                h1 = h1.next

            if not h2:
                h2 = headA
            else:
                h2 = h2.next
        return h1


def removeNthFromEnd(head, n):
    if not head: return head
    p1 = p2 = head
    pre = None
    while p1:
        while n:
            p1 = p1.next
            n -= 1
        if not p1:
            return p2.next
        pre = p2 
        p1 = p1.next
        p2 = p2.next
    pre.next = p2.next
    p2.next = None
    return head


def reorderList(head):
    if not head or not head.next: return head
    def findMid(h):
        if not h:
            return h
        slow = head
        fast = head
        pre = None
        while fast and fast.next:
            pre = slow
            slow = slow.next
            fast = fast.next.next
        if pre:
            pre.next = None
        return slow
    def reverse(h):
        if not h or not h.next: return h
        succ, pre = None, None
        cur = h
        while cur:
            pre = cur
            cur = cur.next
            pre.next = succ
            succ = pre
        return succ
    first = head
    second = reverse(findMid(head))
    d = cur = ListNode(-1)
    while first and second:
        cur.next = first
        cur = cur.next
        first = first.next

        cur.next = second
        cur = cur.next
        second = second.next
    if first or second:
        cur.next = first or second
    return d.next


def mergeKLists(lists):
    if not lists: return None
    def merge(h1, h2):
        d = cur = ListNode(-1)
        while h1 and h2:
            if h1.val < h2.val:
                cur.next = h1
                h1 = h1.next
            else:
                cur.next = h2
                h2 = h2.next
            cur = cur.next
        if h1 or h2:
            cur.next = h1 or h2
        return d.next

    cur = lists[0]  
    for i in range(len(lists)):
        if i > 0: cur = merge(cur, lists[i])
    return cur


def deleteDuplicates(head):
    dummy = pre = ListNode(-1)
    dummy.next = head
    cur = head
    while cur:
        while cur.next and cur.val == cur.next.val:
            cur = cur.next
        if pre.next != cur: # meaning repeating nodes 
            pre.next = cur.next
        else:
            pre = cur
        cur = cur.next
    return dummy.next


def sortedListToBST(head):
    def findMid(head):
        slow = fast = head
        pre = None
        while fast and fast.next:
            pre = slow
            slow = slow.next
            fast = fast.next.next
        if pre:
            pre.next = None
        return slow
    def helper(head):
        if not head: return None
        if not head.next: return TreeNode(head.val)
        mid = findMid(head)
        root = TreeNode(mid.val)
        root.left = helper(head)
        root.right = helper(mid.next)
        return root
    return helper(head)


def sortList(head):
    def merge(h1, h2):
        dummy = ListNode(-1)
        d = dummy
        while h1 and h2:
            if h1.val < h2.val:
                dummy.next = h1
                h1 = h1.next
            else:
                dummy.next = h2
                h2 = h2.next
            dummy = dummy.next
        if h1 or h2:
            dummy.next = h1 or h2
        return d.next

    if not head or not head.next: return head
    slow, fast = head, head
    pre = None 
    while fast and fast.next:
        pre = slow 
        slow = slow.next
        fast = fast.next.next 
    pre.next = None 
    head1 = head
    head2 = slow
    l1 = sortList(head1)
    l2 = sortList(head2)
    return merge(l1, l2)


def swapPairs_iter(head):
    d = ListNode(-1)
    d.next = head 
    run = d
    while run.next and run.next.next:
        first = run.next
        second = run.next.next 
        first.next = second.next
        run.next = second
        run.next.next = first
        run = run.next.next
    return d.next


def swapPairs_recur(head):
    if not head or not head.next: return head
    n = head.next
    head.next = swapPairs_recur(head.next.next)
    n.next = head
    return n


def reverseKGroup(head, k):
    def reverse(h1, h2):
        next = None
        while h1 != h2:
            tmp = h1
            h1 = h1.next
            tmp.next = next
            next = tmp
        return next
    dummy = ListNode(-1)
    d = dummy 
    dummy.next = head
    start, end = head, head
    pre = None
    size = 1 
    while True:
        count = 0
        while count != k and end: 
            pre = end
            end = end.next
            count += 1
            size += 1 
        if count < k and not end: break
        tmp = reverse(start, end)
        d.next = tmp 
        start.next = end
        d = start
        start = end
    return dummy.next 

# Input: "()[]{}"
# Output: true
# {([])}
""" Basic String Manipulation """

def buddy_string(a, b):
    if len(a) != len(b):
        return False
    if a == b:
        return len(set(a)) != len(b)
    count = 0
    idx = []
    for i in range(len(a)):
        if a[i] != b[i]:
            idx.append(i)
            count += 1
        if count > 2:
            return False
    if count < 2:
        return False
    return a[idx[0]] == b[idx[1]] and a[idx[1]] == b[idx[0]]

# print(buddy_string('aabc', 'abac'))

def group_anagram():
    pass 

""" stack """
def valid_parenthesis(s):
    op = '([{'
    cl = ')]}'
    stack = []
    for c in s:
        if c in op:
            stack.append(c)
        if c in cl:
            if not stack:
                return False
            top = stack.pop()
            if cl.index(c) != op.index(top):
                return False
    return len(stack) == 0


# decreasing sequence
def trap_stack(height):
    decr_stack, area, i = [], 0, 0
    while i < len(height):
        # only when insert element when stack is empty or cur less than top element 
        if not decr_stack or height[i] < height[decr_stack[-1]]:
            decr_stack.append(i)
            i += 1
        else: 
            # meaning we encounter first element that is violate the decresing sequence 
            valley = decr_stack.pop()
            if decr_stack:
                area += (min(height[i], height[decr_stack[-1]]) - height[valley]) * (i - decr_stack[-1] - 1)
            # if stack is empty after valley pop, meaning nothing is on the left side of this valley, so there will be 
            # not water trapped. 
    return area
            
# decresing sequence 
def nextGreaterElement(nums1, nums2):
    decr_stack, res = [], [-1]*len(nums1)
    dic = {}
    for n in nums2: 
        while decr_stack and decr_stack[-1] < n: #peek
            dic[decr_stack.pop()] = n #pop
        decr_stack.append(n)
    
    for i, n in enumerate(nums1):
        if n in dic: 
            res[i] = dic[n]
    return res 
        

def nextGreaterElements(nums):
    decr_stack, res = [], [-1]*len(nums)
    for i in range(len(nums)*2):
        while decr_stack and (nums[decr_stack[-1]] < nums[i % len(nums)]):
            res[decr_stack.pop()] = nums[i % len(nums)]
        decr_stack.append(i % len(nums))
    return res


def dailyTemperatures(T):  
    stack = []  # index of decreasing sequence
    res = [0]*len(T)
    for i in range(len(T)):
        while stack and T[stack[-1]] < T[i]:
            cur = stack.pop()
            res[cur] = i - cur
        stack.append(i)
    return res

# forming an increasing sequence stack 
def largestRectangleArea(heights):
    stack, maxArea = [], 0
    for i in range(len(heights)+1):
        h = heights[i] if i < len(heights) else 0
        while stack and heights[stack[-1]] >= h:
            cur_height = heights[stack.pop()]
            width = i - stack[-1] - 1 if stack else i
            maxArea = max(maxArea, cur_height*width)
        stack.append(i)
    return maxArea


# too diao
def maximalRectangle(matrix):
    if not matrix or not matrix[0]:return 0
    maxArea, n = 0, len(matrix[0])
    heights = [0]*(n+1)
    # 2d -> 1d
    # row by row check largest area.
    for row in matrix:
        for i in range(n):
            heights[i] = heights[i]+1 if row[i] == '1' else 0
        # this block of code has to be inside of outer for loop so it can avoid edge case like [[1,0], [0,1]] which area is 0. if you put following block outside, will return  1
        stack = []
        for i in range(n+1):
            height = heights[i] if i < n+1 else 0
            while stack and heights[stack[-1]] >= height:
                cur_height = heights[stack.pop()]
                width = i - stack[-1] - 1 if stack else i
                maxArea = max(maxArea, cur_height * width)
            stack.append(i)
    return maxArea

# increasing sequence stack 
def removeDuplicateLetters(s):
    stack, dic = [], {}
    for c in s: 
        dic[c] = dic.get(c, 0)+1

    for c in s: 
        dic[c] -= 1
        # if c already in stack, no need to pop
        if c in stack: continue # handle edge case 'abacb'
        while stack and ord(c) < ord(stack[-1]) and dic[stack[-1]] > 0:
            stack.pop()
        stack.append(c)
    return ''.join(stack)


# 23145 2
# maintain a increaing sequence in stack, the top of the stack is the biggest digit
# the res of stack is smaller and at weighed more position
# if you find a digit that is greater than top, pop till the top is less than this digit
# at the same time maintain the condition that will stop the popping
def remove_Kdigits(num, k):
    stack = []
    for d in num:
        while stack and stack[-1] > d and k > 0:
            stack.pop()
            k -= 1
        stack.append(d)
    return ''.join(stack)
# print(remove_Kdigits('23145', 2))


""" BFS-Board and DFS and Dijkstra """
# bfs
def hasPath(maze, start, destination):
    directions = [(1, 0), (-1, 0), (0, 1), (0,-1)] 
    queue = [start]
    n , m = len(maze), len(maze[0])
    while queue:
        # every position in queue is where ball stops (border or 1)
        i, j = queue.pop(0) # using python list to simulate queue. pop front. unpacking 
        # set visited positon value to -1
        maze[i][j] = -1 
        if i == destination[0] and j == destination[1]: return True
        for x, y in directions:
            row = i + x
            col = j + y
            # loop until hit the wall or boarder
            while 0 <= row < n and 0 <= col < m and maze[row][col] != 1:
                row += x
                col += y
            # a step back
            row -= x 
            col -= y
            if maze[row][col] == 0 and [row, col] not in queue: 
                queue.append([row, col])
    return False 


from collections import heapq
def shortestDistance(maze, start, destination):
    pq = [(0, start[0], start[1])]
    directions = [(1, 0), (-1, 0), (0, 1), (0,-1)] 
    n , m = len(maze), len(maze[0])
    while pq: 
        count, i, j = heapq.heappop(pq)
        if maze[i][j] == -1: continue 
        maze[i][j] = -1 
        if i == destination[0] and j == destination[1]: return count
        for x, y in directions:
            row = x + i
            col = y + j 
            local = 1 
            while 0 <= row < n and 0 <= col < m and maze[row][col] != 1:
                row += x 
                col += y 
                local += 1
            row -= x 
            col -= y 
            local -= 1
            # check out tuple comparison. basic idea is compare item by item lexigraphically. 
            heapq.heappush(pq, (count+local, row, col)) # maintian min heap, the shortest path always on the root. 
    return -1 

# bfs 
def updateMatrix(matrix):
    queue = []
    n, m = len(matrix), len(matrix[0])
    for i in range(len(n)):
        for j in range(m):
            if matrix[i][j] != 0: 
                matrix[i][j] = float('inf') # make sure all 1's get updated and added to queue
            else:
                queue.append((i, j))
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    while queue:
        cur_row, cur_col = queue.pop(0)
        for x, y in directions: 
            row = cur_row + x
            col = cur_col + y
            if 0 <= row < n and 0 <= col < m:
                # if prev update is not the shortest, update it with shorter distance. 
                if matrix[row][col] > matrix[cur_row][cur_col] + 1: 
                    matrix[row][col] = matrix[cur_row][cur_col] + 1
                    queue.append((row, col))
    return matrix


def updateMatrix_dfs(matrix):
    n, m = len(matrix), len(matrix[0])
    for i in range(n):
        for j in range(m):
            if matrix[i][j] == 1:
                matrix[i][j] = float('inf')

    def dfs(matrix, i, j, n, m, dist):
        # matrix[i][j] can only be 0, inf, or some visited position 
        # if matrix[i][j] is 0 or inf then matrix[i][j] < dist is false, when visited position less than dist, return meaning 
        # right value is in place. 
        if i < 0 or i >= n or j < 0 or j >= m or matrix[i][j] < dist:
            return 
        matrix[i][j] = dist
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        for x, y in directions:
            dfs(matrix, i+x, j+y, n, m, dist+1)

    for i in range(n):
        for j in range(m):
            if matrix[i][j] == 0:
                dfs(matrix, i, j, n, m, 0)
    return matrix


def wallsAndGates(rooms):
    if not rooms: return rooms
    queue = []
    n, m = len(rooms), len(rooms[0])
    for i in range(n):
        for j in range(m):
            if rooms[i][j] == 0:
                queue.append((i, j))

    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    while queue:
        cur_row, cur_col = queue.pop(0)
        for x, y in directions:
            row = cur_row + x
            col = cur_col + y
            if 0 <= row < n and 0 <= col < m and rooms[row][col] != -1:
                if rooms[row][col] > rooms[cur_row][cur_col] + 1:
                    rooms[row][col] = rooms[cur_row][cur_col] + 1
                    queue.append((row, col))
    return rooms



def floodFill(image, sr, sc, newColor):
    if image[sr][sc] == newColor: return image
    def dfs(image, i, j, old_color, new_color): 
        if i < 0 or i >= len(image) or j < 0 or j >= len(image[0]) or image[i][j] != old_color:
            return
        image[i][j] = new_color
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        for x, y in directions:
            dfs(image, i+x, j+y, old_color, new_color)

    dfs(image, sr, sc, image[sr][sc], newColor)
    return image


def longestIncreasingPath(matrix):
    pass


def pacificAtlantic(matrix):
    pass


# Dijkstra algorithm see TCRC 
def networkDelayTime(times, N, K):
    # use defaultdict with list factory meaning the value of given key will be store in a list
    # [(0, k)] (estimate, node)
    # s is for storing shortest path to each node 
    pq, s, adj = [(0, K)], {}, collections.defaultdict(list) 
    for u, v, w in times:
        # representing edges
        adj[u].append((w, v))

    while pq:
        time, node = heapq.heappop(pq)
        if node not in s:
            s[node] = time  # shortest path to this node from source
            for w, v in adj[node]:
                heapq.heappush(pq, (w+time, v))
    return max(s.values()) if len(s) == N else -1


def findCheapestPrice(n, flights, src, dst, K):
    pq, edges = [(0, src, 0)], collections.defaultdict(list)
    for u, v, w in flights: 
        edges[u].append((w, v))
    
    while pq: 
        price, node, stops = heapq.heappop(pq)
        if node == dst: return price
        if stops <= K: 
            # add all current node's neibours to the queue, basically like bfs(use normal queue) but with priorityqueue instead
            for p, n in edges[node]:
                heapq.heappush(pq, ((p+price), n, stops+1))
    return -1
        


""" DP """
def trap_dp(height):
    pass 

""" Math """

""" Binary Search """

""" Hashtable """

""" Bit Manipulation """
def subsets_bit(nums):
    pass

""" priority queue and heap """
def findKthLargest(nums, k):
    pass

def topKFrequent(nums, k):
    pass 


def kthSmallest(matrix, k):
    pass

""" Tree """
# class TreeNode(object):
#     def __init__(self, val):
#         self.val = val
#         self.left = None
#         self.right = None


# class ListNode(object):
#     def __init__(self, val):
#         self.val = val
#         self.next = None
# root = TreeNode(1)
# root.left = TreeNode(2)
# root.right = TreeNode(3)
# root.left.left = TreeNode(4)
# root.left.right = TreeNode(5)
# root.right.left = TreeNode(6)
# root.right.right = TreeNode(7)

# output: 4251637
def in_order(r):
    stack = []
    res = []
    while r or stack:
        while r:
            stack.append(r)
            r = r.left
        if stack:
            r = stack.pop()
            res.append(r.val)
            r = r.right
    return res
# print(in_order(root))          

# output: 1245367 
def pre_order(r):
    stack = [r]
    res = []

    while stack: 
        top = stack.pop()
        res.append(top.val)
        if top.right: 
            stack.append(top.right)
        if top.left: 
            stack.append(top.left)
    return res 
# print(pre_order(root))
  
#  1
# /\
# 2 3
# /\
# 4 5

# 4526731
def post_order(r):
    stack, res = [r], []
    while stack:
        top = stack.pop()
        res.append(top.val)
        if top.left:
            stack.append(top.left)
        if top.right: 
            stack.append(top.right)

    return res[::-1]
# print(post_order(root))


# head = ListNode(1)
# head.next = ListNode(2)
# head.next.next = ListNode(3)
# head.next.next.next = ListNode(4)


# iter
# 1-2-3-4
def reverseLL_iter(head): 
    pre, cur = None, None
    while head:
        pre = head
        head = head.next
        pre.next = cur
        cur = pre
    return cur

# head = reverseLL_iter(head)
# while head: 
#     print(head.val)
#     head = head.next

def reverseLL_recur(head):
    if not head.next: return head
    new_head = reverseLL_recur(head.next)
    head.next.next = head
    head.next = None
    return new_head


# head = reverseLL_recur(head)
# while head:
#     print(head.val)
#     head = head.next


def isSymetric_recur(r):
    def helper(r1, r2):
        if not r1 and not r2: return True
        if not r1 or not r2: return False
        return r1.val == r2.val and helper(r1.left, r2.right) and helper(r1.right, r2.left)
    return helper(r, r)

#   1
#  /\
# 2  2
# /\ /\
# 1 2 2 1
def isSymetric_iter(r):
    def helper(r1, r2):
        stack = [r2, r1]
        while stack: 
            r1 = stack.pop()
            r2 = stack.pop()
            # the order of following two line can be swapped. if swappped will return false when both are null nodes
            if not r1 and not r2: continue
            if not r1 or not r2: return False
            if r1.val != r2.val: return False
            stack.append(r2.right)
            stack.append(r1.left)
            stack.append(r2.left)
            stack.append(r1.right)
        return True
    return helper(r, r)


# print(isSymetric_iter(root))
# print(isSymetric_recur(root))
