"""
Author: Eddie Dong
Description: 
Solutions for leetcode problems in Python
Cover most of topics of fundatmental algorithms and data structures.
"""

# command k, command 0 to fold all codes 
import string
from itertools import product
import random
import heapq
import collections
from collections import deque
from typing import *

""" prefix sum problems"""
# 560
# similar to technique used in two sum. Use hashmap
def subarraySum(self, nums: List[int], k: int) -> int:
    cnt = 0
    dic = {0: 1} 
    prefix = 0
    for i, n in enumerate(nums):
        prefix += n
        if prefix - k in dic:
            cnt += dic[(prefix-k)]
        dic[prefix] = dic.get(prefix, 0)+1
    return cnt
      


""" K sum problems """
# 1
# conclusion: when we see sorted array, we need to think of two pointers and binary search
def twoSum(nums, target):
    m = {} # [k=comp, v=index] store complememt of later elements 
    for i, v in enumerate(nums):
        if v not in m: 
            m[target - v] = i
        return [m[v], i]


# 15
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


# 16
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



# 18
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


# 325 
# store prefix sum and index in hashtable
# store only the first index of a prefix sum only, to make the distance maximal.
# for i < j, prefix(j) - prefix(i) == k, the length of the subarray sum to k, is i-j, which not include i
# thus we initialize mp = {acc: -1}, maybe the subarray is starting from begining to j not include i, which is -1
# T:O(n), S:O(n)
def maxSubArrayLen(self, nums: List[int], k: int) -> int:
    acc, ans = 0, 0
    # whenever acc = 0 in your subarray, the value at mp[0] remains -1 instead of being changed to the current index, i.
    mp = {acc: -1} 
    for i in range(len(nums)):
        acc += nums[i]
        # only first time meet gets stored
        if acc not in mp:
            mp[acc] = i
        if acc-k in mp:
            ans = max(ans, i-mp[acc-k])
    return ans
    


# 653
# use hashset does not utilize the bst, alternatively, can do inorder traverse then two pointers to  find the pair.
def findTarget(self, root: TreeNode, k: int) -> bool:
    return self.dfs(root, set(), k)
    
# add each subtree's complement into a set, check if current node value existed in the set. 
def dfs(self, root, seen, k):
    if not root: return False
    if root.val in seen: 
        return True
    seen.add(k-root.val)
    return self.dfs(root.left, seen, k) or  self.dfs(root.right, seen, k)
        
        
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
        # for this implementation we use last position as pivot index 
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
# merge sort recursive
def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    left = merge_sort(arr[ :len(arr)//2])
    right = merge_sort(arr[len(arr)//2: ])
    return merge(left, right)

# merge two sorted arr
# if we merge a linkedlist, the space complexity would be O(1)
def merge(arr1, arr2):
    res = []
    i = j = 0
    n, m = len(arr1), len(arr2)
    while i < n and j < m:
        if arr1[i] < arr2[j]:
            res.append(arr1[i])
            i += 1
        else:
            res.append(arr2[j])
            j += 1
    res += arr1[i:] + arr2[j:]
    return res 
       

""" 
heap sort
"""
class MaxHeapBase(object):
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
        


""" 
cons: O(n^2)
pros: only make O(n) swap, it is good for memory writes are costly operation and only user constant space
"""
def selection_sort(arr):
    for i in range(len(arr)):
        min_idx = i
        for j in range(i, len(arr)):
            if arr[j] < arr[min_idx]:
                min_idx = j
        arr[i], arr[min_idx] = arr[min_idx], arr[i]
    print(arr)


"""
pros: It is in place and easy to detect small errs by swap two elements at a time
cons: normal bubble sort will have worse and average time o(n^2) even when the input is sorted
"""
# a bit optimized. when array is sorted, now takes O(n)
def bubble_sort_clean(arr):
    # for all elements, i is the last position to be compared with previous position
    for i in range(len(arr)-1, 0, -1):
        for j in range(i):
            if arr[j] > arr[j+1]:
                 arr[j], arr[j+1] = arr[j+1], arr[j]
    print(arr)

"""
pros: when input is small, it's quite efficient and reduced swaps and stable
cons: when input is larget, its slow 
"""
# reduce number of swap operations
def insertion_sort_clean(arr):
    for i in range(1, len(arr)):
        pos = i
        cur_val = arr[pos]
        # shifing the elements to the right
        while pos > 0 and arr[pos-1] > cur_val:
            arr[pos] = arr[pos-1]
            pos -= 1
        arr[pos] = cur_val
    print(arr)



""" Interval problems """
# 56
def merge_(self, intervals):
    out = [] # stores non overlaping intervals
    # sort the arrray using according to [0]
    sorted_intervals = sorted(intervals)
    for inter in sorted_intervals:
        # compare current interval with the existing interval,and if current [0] is <= [1] we need to update out[1]
        if out and out[-1][1] >= inter[0]:
            out[-1][1] = max(out[-1][1], inter[1])
        else:
            out.append(inter)      
    return out


# 57
def insert(self, intervals, newInterval):
    out = []
    intervals = sorted(intervals + [newInterval])
    for inter in intervals:
        if out and out[-1][1] >= inter[0]:
            out[-1][1] = max(out[-1][1], inter[1])
        else:
            out.append(inter)
    return out



# 252
# same as above 
def canAttendMeetings(self, intervals):
    out = []
    for inter in sorted(intervals):
        if out and out[-1][1] > inter[0]:
            return False
        else:
            out.append(inter)
    return True


# slight modify
def canAttendMeetings_(self, intervals):
    intervals = sorted(intervals)
    for i in range(1, len(intervals)):
        if intervals[i][0] < intervals[i-1][1]:
            return False
    return True 



# 253
# think case[[1,2], [3,4], [5,6]]. Given start < end and if only one interval, only needs one room. 
def minMeetingRooms(self, intervals):
    starts = sorted(i[0] for i in intervals)
    ends = sorted(i[1] for i in intervals)
    need, e = 0, 0 
    for s in range(len(starts)):
        if starts[s] < ends[e]:
            need += 1
        else:
            e += 1
    return need

# min heap store end time, sort the start time. return the lenght of heap.
#  better
def minMeetingRooms(self, intervals: List[List[int]]) -> int:
    intervals.sort(key=lambda x: x[0])
    pq = []
    rooms = 0
    for interval in intervals:
        if not pq:
            heapq.heappush(pq, interval[1])
            rooms += 1 
        else:
            if pq[0] <= interval[0]:
                heapq.heappop(pq)
            else:
                rooms+= 1 
            heapq.heappush(pq, interval[1])
    return rooms
        

# 435
# intuition: sort according to the ends, because the smallest end will leave the most space to the rest, and if overlap remove it. 
def eraseOverlapIntervals(self, intervals):
    end = float('-inf')
    intervals = sorted(intervals, key=lambda i : i[1])
    rm = 0 
    # in an interval. start < end
    for inter in intervals:
        if end <= inter[0]:
            end = inter[1]
        else:
            rm += 1
    return rm



# 986
# two pointers, just merge two sorted arrays.
def intervalIntersection(self, A, B):
    p, q = 0, 0
    n, m = len(A), len(B)
    out = []
    while p < n and q < m:
        if not (A[p][1] < B[q][0] or A[p][0] > B[q][1]):
            out.append([max(A[p][0], B[q][0]), min(A[p][1], B[q][1])])
            if A[p][1] < B[q][1]:
                p += 1
            else:
                q += 1
        elif A[p][1] < B[q][0]:
            p += 1 
        else:
            q += 1  
    return out


# 452
# intersaction.
def findMinArrowShots(self, points: List[List[int]]) -> int:
    points = sorted(points, key=lambda i : i[1])
    pre, cnt = None, len(points)
    for p in points:
        if pre and pre[1] >= p[0]:
            pre = [max(p[0], pre[0]), min(pre[1], p[1])]
            cnt -= 1
        else:
            pre = p
    return cnt
        


""" sliding window, substring"""
# 727 
# tip: strip unnecessary prefix.
def minWindow(self, S: str, T: str) -> str:
    s_idx = t_idx = 0
    start = -1
    n, m = len(S), len(T)
    min_len = n
    
    while s_idx < n:
        if S[s_idx] == T[t_idx]:
            t_idx += 1
            # going backward 
            if t_idx == m:
                end = s_idx + 1
                t_idx -= 1
                while t_idx >= 0:
                    while S[s_idx] != T[t_idx]:
                        s_idx -= 1
                    s_idx -= 1
                    t_idx -= 1
                
                t_idx += 1
                s_idx += 1
            
                if end - s_idx < min_len:
                    min_len = end - s_idx
                    start = s_idx
        s_idx += 1
    return '' if start == -1 else S[start : start + min_len]
            

# 75
def sortColors(self, nums):
    n = len(nums)
    p0, cur = 0, 0 
    p2 = n-1
    while cur <= p2:
        if nums[cur] == 0:
            nums[cur], nums[p0] = nums[p0], nums[cur]
            p0 += 1
            cur += 1
        elif nums[cur] == 2:
            nums[cur], nums[p2] = nums[p2], nums[cur]
            p2 -= 1
        else:
            cur += 1


# 26
def removeDuplicates(self, nums):
    j = 1 #first position will never available, thus we start from second
    for i in range(1,len(nums)):
        # becaue the array is sorted, so the compare two closed element will find the unique ones
        if nums[i] != nums[i-1]:
            #we just assign the unique element to current available position, we dont do swap becuase we dont care about 
            # current available position element we just replace it. 
            nums[j] = nums[i]
            j += 1
    return j


# 438
# with for loop implmentation
def findAnagrams(self, s, p):
    dic_p, dic_s = {}, {}
    for c in p:
        dic_p[c] = dic_p.get(c, 0) + 1
    j, res = 0, []
    for i, ch in enumerate(s):
        dic_s[ch] = dic_s.get(ch, 0) + 1
        if i - j + 1 == len(p):
            if dic_p == dic_s:
                res.append(j)
            dic_s[s[j]] -= 1 
            if dic_s[s[j]] == 0:
                del(dic_s[s[j]])
            j += 1
    return res


# optimal solution. w/o using dic and del similar to below probelm
def findAnagrams(self, s: str, p: str) -> List[int]:
    target = [0] * 26
    for c in p:
        target[ord(c) - ord('a')] += 1 
    
    window, res = [0]*26, []
    for i, c in enumerate(s):
        window[ord(c) - ord('a')] += 1 
        
        if i - len(p) >= 0:
            window[ord(s[i-len(p)]) - ord('a')] -= 1 
        
        if window == target:
            res.append(i-len(p)+1)
    return res



# 567
# sliding window, maintain a end - begin + 1 == len(s1) window 
def checkInclusion(self, s1, s2):
    dic1, dic2 = {}, {}
    for c in s1:
        dic1[c] = dic1.get(c, 0) + 1
    
    begin = 0 
    for end, c in enumerate(s2):
        dic2[c] = dic2.get(c, 0) + 1
        if dic1 == dic2: 
            return True 
        # currently dic1 and dic2 are not the same, and length of current substring is the same as the s1, 
        # we need to shrink the window and remove not matched entr
        # dont write the condition as len(dic1) == len(dic2) becuase, when two dics are different, their represented substring could be same length, 
        # becuase two dics are difference can cause by key difference or value difference
        if end - begin + 1 == len(s1):
            dic2[s2[begin]] -= 1
            if dic2[s2[begin]] == 0:
                del(dic2[s2[begin]])
            begin += 1
    return False 


# optimal solution: without using dictionary and left window is represented using subtraction of window size. 
def checkInclusion(self, s1: str, s2: str) -> bool:
    target = [0] * 26
    # s1 char count mapping
    for c in s1:
        idx = ord(c) - ord('a')
        target[idx] += 1 
    
    window = [0] * 26 
    for i, c in enumerate(s2):
        idx = ord(c) - ord('a')
        window[idx] += 1
        
        if i - len(s1) >= 0:
            window[ord(s2[i-len(s1)]) - ord('a')] -= 1
        
        if window == target:
            return True
    return False
                



# 763
# greedy + two pointers
def partitionLabels(S):
    # last occurence of each character
    dic = {c:i for i, c in enumerate(S)} # fast way to create dictionary
    j = anchor = 0
    res = []
    for i, c in enumerate(S):
        # extend to the farthest partition index
        j = max(j, dic[c])
        # if current index equals the farthest partition index, meaning current substring from [anchor to j] form a valid substring
        if i == j: 
            res.append(j - anchor + 1)
            anchor = i + 1 
    return res 


# 713
# always maintain a window that is less than k, then count number of qualified subsets between i and j
def numSubarrayProductLessThanK(nums, k):
    if k <= 1: return 0
    prod, count, i = 1, 0, 0
    n = len(nums)
    for j in range(n):
        prod *= nums[j]
        while prod >= k:
            prod //= nums[i]
            i += 1
        """ every add one element into the product for example [abc] add d into the product, we have [abcd], [bcd], [cd], [d]
        that is equal to total count of element after adding d which is i-j+1 """
        count += j - i + 1
    return count


def numSubarrayProductLessThanK(self, nums: List[int], k: int) -> int:
    prod = 1
    total= j = 0
    for i, n in enumerate(nums):
        prod *= n
        while j <= i and prod >= k:
            prod //= nums[j]
            j += 1
        total += i-j+1
    return total
            
            
            


# 209
def minSubArrayLen(self, s, nums):
    min_len = len(nums)
    total = j = 0 
    entered = False
    
    for i in range(len(nums)):
        total += nums[i]
        
        while total >= s:
            entered = True 
            min_len = min(min_len, i - j + 1)
            total -= nums[j]
            j += 1
        
    return min_len if entered else 0 


# find the 临界点 , find the min subarray that less than s, but if you add [i] to it, it will be >= than s
def minSubArrayLen_(self, s, nums):
    min_sub = float('inf')
    j, total = 0, 0
    for i in range(len(nums)):
        total += nums[i]
        if total >= s:
            while j <= i and total >= s:
                total -= nums[j]
                j += 1
            min_sub = min(min_sub, i-j+1)
    return min_sub+1 if min_sub != float('inf') else 0 


   
#862
# different with above. negative number is allowed. using monoqueue to solve it. 
# using deque to implement sliding window with begin skipping steps. 
""" 
because the array may contain negative numbers, thus we can use prefix sum to solve this problem. 
we use deque to store the increasing prefix sum index.
-reason to use deque: we constantly add and remove from back and front of queue, thus we need a fast ds to to this. 
-reason to maintain increasing sequence because if cur_sum - [deque[0]] < k, then the [deqeue[1]] or later must not make it >= k

if total - q[0][1] >= K, we can shrink the window, to find the min. 
"""
def shortestSubarray(self, A, K):
    q = collections.deque([[0, 0]]) #careful with the initalization, its is list of list
    res = float('inf')
    total = 0 
    
    for i, v in  enumerate(A):
        total += v
        
        while q and total - q[0][1] >= K: 
            res = min(res, i + 1 - q.popleft()[0])
        
        while q and total < q[-1][1]: 
            q.pop()
        
        q.append([i+1, total])
    
    return res if res < float('inf') else -1
    


# monotonic queue
# 239 
# double ended queue. 
def maxSlidingWindow(nums, k):
    dq = collections.deque()
    res = []
    for i in range(len(nums)):
        # see if current max in dq is inside the window
        if dq and dq[0] == i - k: 
            dq.popleft()
        # maintain decreasing queue
        while dq and nums[dq[-1]] < nums[i]: 
            dq.pop()
        dq.append(i)
        # initially, element is less than k
        if i - k + 1 >= 0:
            res.append(nums[dq[0]])
    return res 


# easy understand version sliding window using double ended queue
def maxSlidingWindow(nums, k):
    deque = collections.deque()
    res = []
    j = 0 
    for i in range(len(nums)):
        while deque and deque[0] < j:
            deque.popleft()
            
        while deque and nums[deque[-1]] < nums[i]:
            deque.pop()
        deque.append(i)
        if i - j + 1 == k:
            res.append(nums[deque[0]])
            j += 1
    return res



# 76
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


# optimal solution
# use array to replace dictionary. 
# positive: needs, negative: extra or not needed, zero: just meets the need
def minWindow(self, s: str, t: str) -> str:
    t_map = [0] * 128
    for c in t: t_map[ord(c)] += 1 
    j, res = 0, ''
    need, length = len(t), len(s)
    for i, c in enumerate(s):
        if t_map[ord(c)] > 0:
            need -= 1 
        t_map[ord(c)] -= 1
        while need == 0:
            if i - j + 1 <= length:
                length = i - j + 1
                res = s[j: i+1]
            t_map[ord(s[j])] += 1
            if t_map[ord(s[j])] > 0: need += 1 
            j += 1 
    return res



# 3
def lengthOfLongestSubstring(s):
    dic = {}
    left = max_len = 0
    for right in range(len(s)):
        if s[right] in dic and dic[s[right]] >= left:
            left = dic[s[right]]+1
        dic[s[right]] = right
        max_len = max(max_len, right-left+1)
    return max_len



# 159
# increment the left pointer when the size of the window greater than 2 
# when the size equal to 2 update the result
# cannot use set to do this problem becuse set does not allow dup, so we can use frequency dict to reduce the frequence if it has more than one in the string.
def lengthOfLongestSubstringTwoDistinct(self, s):
    j, leng = 0 , 0  
    dic = {}
    for i, ch in enumerate(s):
        dic[ch] = dic.get(ch, 0) + 1
        while len(dic) > 2: 
            dic[s[j]] -= 1 
            if dic[s[j]] == 0: 
                del(dic[s[j]])
            j += 1
        leng = max(leng, i - j + 1)
    return leng 


# solution without deleting entry in the dictionary
def lengthOfLongestSubstringTwoDistinct(self, s: str) -> int:
        chars = {}
        left = longest = 0 
        numOfChars = 0 
        for right in range(len(s)):
            if s[right] not in chars or chars[s[right]] == 0: 
                numOfChars += 1 
                
            chars[s[right]] = chars.get(s[right], 0 ) + 1 
            
            while numOfChars > 2:
                chars[s[left]] -= 1
                if chars[s[left]] == 0: 
                    numOfChars -= 1
                left += 1
            longest = max(longest, right - left + 1)
        return longest


# 424 
""" we only care about the len of the valid substring, so the window either expand right by adding one char to the right, or shift to the right by one 
it never shrinks. the max_count is historical max_count that forms the valid subtring """
# two pointer/sliding window
# generally, when k is enough, window is expanding by adding char on the right, when k is not enough, we just shift the entire window to the right by 1 step
# the tricky idea here is that the we never have to update max_freq if max_freq char is being left out due to shifting of left pointer. 
# and when we find a valid max window, we never shrink, we just shift entire window to the right by 1. 
# here the max_freq is prepresenting a historical max_freq that forms the previous max_window.
# we only expand the window when some char count is greater than previous max_freq (we will not go into the inner if block, thus we not shifting right)

# key: find the max of most frequent char in a valid window 
def characterReplacement(self, s, k):
    j, ans = 0, 0
    chars = {}
    max_freq = 0
    for i, ch in enumerate(s):
        chars[ch] = chars.get(ch, 0) + 1
        max_freq = max(max_freq, chars[ch])
        if i - j + 1 - max_freq > k:
            chars[s[j]] -= 1
            j += 1
        ans = max(ans, i-j+1)
    return ans


# use array
def characterReplacement(self, s: str, k: int) -> int:
    dic = [0] * 26
    longest = j = most_common = 0
    for i, c in enumerate(s):
        idx = ord(c) - ord('A')
        dic[idx] += 1
        most_common = max(most_common, dic[idx])
        if i - j + 1 > most_common + k:
            dic[ord(s[j]) - ord('A')] -= 1 
            j += 1
        longest = max(longest, i - j + 1)
    return longest



""" monotonic stack/queue"""
# 896 
# one pass. 
# update two indicators along the way. either is true is true. 
def isMonotonic(self, A: List[int]) -> bool:
    incr = decr = 1
    for i in range(1, len(A)):
        incr *= (A[i] >= A[i-1])
        decr *= (A[i] <= A[i-1])
    return incr or decr
        
# 255
# intuition: it's preorder, so there should be some decreasing order, once some number breaks the order, this number must be on some node's right subtree.

# push left subtree on the stack and once a number is breaking the decreasing order, pop until this number is less than the top of the stack
# which means, this number is on right subtree of its highest ancestor, 
# the next number must at least greater than the lowest ancestor and less then top of the stack to be placed on the right subtree
# if next number if greater than the top of the stack, you need find its lowest ancestor again because it could be greater than the root. 

def verifyPreorder(self, preorder: List[int]) -> bool:
    decre_stack = []
    highest_ancestor = float('-inf')
    for v in preorder:
        if v < highest_ancestor: return False
        while decre_stack and v > decre_stack[-1]:
            lowest_ancestor = decre_stack.pop()
        decre_stack.append(v)
    return True
        


# 735
# maintain a postive stack.
def asteroidCollision(self, ast: List[int]) -> List[int]:
    stack, res = [], []
    for a in ast:
        if not stack and a < 0: res.append(a)
        d = a 
        while stack and d < 0: 
            if stack[-1] < abs(d):
                stack.pop()
                if not stack: res.append(a)
            elif stack[-1] == abs(d):
                stack.pop()
                d = abs(d)
            else:
                d = abs(d)     
        if a >= 0: stack.append(a)  
    return res + stack
            

# other solution, same idea using stack
def asteroidCollision(self, asteroids: List[int]) -> List[int]:
    res, stack = [], []
    for ast in asteroids:
        if ast > 0: 
            stack.append(ast)
        else: 
            if not stack:
                res.append(ast)
            else:
                if stack[-1] + ast <= 0:
                    while True:
                        if not stack: 
                            res.append(ast)
                            break
                        if stack[-1] + ast < 0:
                            stack.pop()
                            continue
                        elif stack[-1] + ast == 0:
                            stack.pop()
                            break
                        else: 
                            break
    return res + stack
                        


# 496
# decresing sequence, next greater means the first met greater.

def nextGreaterElement(nums1, nums2):
    decr_stack, res = [], [-1]*len(nums1)
    dic = {}
    for n in nums2:
        while decr_stack and decr_stack[-1] < n:  # peek
            # assign n to be the key of all the previous values that is less than n.
            dic[decr_stack.pop()] = n  # pop2
        decr_stack.append(n)

    for i, n in enumerate(nums1):
        # if n is not in dic, meaning there is nothing greater than n.
        if n in dic:
            res[i] = dic[n]
    return res


# 503
# visualize the decreasing stack
# \   |
#  \  |
#   \ |
def nextGreaterElements(nums):
    decr_stack, res = [], [-1]*len(nums)
    for i in range(len(nums)*2):
        while decr_stack and (nums[decr_stack[-1]] < nums[i % len(nums)]):
            res[decr_stack.pop()] = nums[i % len(nums)]
        decr_stack.append(i % len(nums))
    return res



# 739
def dailyTemperatures(self, t):
    n = len(t)
    decr_stack, res = [], [0] * n

    for i in range(n):
        while decr_stack and t[decr_stack[-1]] < t[i]:
            k = decr_stack.pop()
            res[k] = i - k
        decr_stack.append(i)

    return res



# 42
# brute force: TLE
# add up how much each bar can trap water. 
# find the highest bar on the left and right side of current bar and the shorter bar of the left and right will decide how much the current bar can trap. 
def trap(self, height: List[int]) -> int:
    total = 0
    for i in range(1, len(height)-1):
        left_max = right_max = -1
        left = right = i 
        while left >= 0:
            left_max = max(left_max, height[left])
            left -= 1 
        
        while right < len(height):
            right_max = max(right_max, height[right])
            right += 1 
        
        total += min(left_max, right_max) - height[i]
    return total



# decreasing sequence
# use elements in the decreasing stack to be the water containers, use the shorter of first longer bar on the left of current valley and the 
# current bar [i] to calculate the hight differences. 
def trap(self, height):
    stack, area = [], 0
    for i, k in enumerate(height):
        while stack and height[stack[-1]] < k:
            valley = stack.pop()
            if stack:
                diff = min(k, height[stack[-1]]) - height[valley]
                width = i - stack[-1] - 1
                area += diff * width
        stack.append(i)
    return area



# two pointers 
def trap(self, height: List[int]) -> int:
    level = total = 0
    left, right = 0, len(height)-1
    while left < right: 
        if height[left] < height[right]: 
            level = max(level, height[left])
            total += level - height[left]
            left += 1 
        else:
            level = max(level, height[right])
            total += level - height[right]
            right -= 1
    return total
                
        

# 84
# forming an increasing sequence stack
""" 
invariant: 
largest histogram area's height should be one of the bars as its height and this bar should always greater than 0, and 
for each bar [i], use this par as the height of the rectangle, find the left and right boundries of the rectangle
use [i] to be the height of a rectangle, the rectangle will expand until left and right bars are less than [i]
"""
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



# 85
# stack
def maximalRectangle(matrix):
    if not matrix or not matrix[0]:
        return 0
    maxArea, n = 0, len(matrix[0])
    heights = [0]*(n+1)
    # 2d -> 1d
    # row by row check largest area.
    for row in matrix:
        for i in range(n):
            heights[i] = heights[i]+1 if row[i] == '1' else 0

        stack = []
        for i in range(n+1):
            height = heights[i] if i < n+1 else 0
            while stack and heights[stack[-1]] >= height:
                cur_height = heights[stack.pop()]
                width = i - stack[-1] - 1 if stack else i
                maxArea = max(maxArea, cur_height * width)
            stack.append(i)
    return maxArea


# 402
# left side is weighed more than right side
# iteration direction:
# remove rule: maintain a increasing sequence and for a increaing sequence, remove the right most digit will generate smallest result because,
# if you remove middle digit, the latter greater digit will shift left making the result greater.
def removeKdigits(self, num, k):
    if len(num) == k:
        return '0'
    stack = []
    # add 1 in case the k is not used up at the last position.
    for i in range(len(num)+1):
        d = num[i] if i < len(num) else '0'
        while stack and stack[-1] > d and k > 0:
            stack.pop()
            k -= 1
        if i < len(num):
            stack.append(d)

    while stack and stack[0] == '0' and len(stack) > 1:
        stack = stack[1:]

    return ''.join(stack)



# 316
# increasing sequence stack
# similar to above problem, using greedy strategy, keep the small dup char remove the larger dup char 
def removeDuplicateLetters(self, s):
    stack, dic = [], {c : i for i, c in enumerate(s)} #last occurence
    visited = set()
            
    for k, ch in enumerate(s): 
        if ch not in visited: 
            # if top element last occurence is behind k, meaning we can delete the dup at positon within [0,k]
            while stack and stack[-1] > ch and k < dic[stack[-1]]: 
                # update the visited when popped off 
                visited.remove(stack.pop())
            
            visited.add(ch)
            stack.append(ch)
    return ''.join(stack)




# 907
"""
core idea here is to use [i] as the min of some subarray, and expand to left and right. in other word, we are trying to the left and right boundries of 
[i] as the min of the [left : right] subarray
thus we use two increasing stacks to find the first number that is less than [i] on the left and right of [i]
then we get count of elements that are greater than or equal to [i] on the left or right of the i, represented by left and right 
their cartitian product left * right are total number of subarrays that has [i] as min, then sum([i]*left*right) will be the result.

handle duplicates: for edge cases such as [1,2,1], we notice that subarray [0:3] and [-1: -3] are the same subarray, thus we handle it by 
using stricter(>=) condition on one of the stacks, the other one using looser(>) condition
"""
def sumSubarrayMins(self, A):
    n = len(A)
    left, right = [0] * n , [0] * n
    s1, s2 = [], []
#  find the first number on the left of [i] 
    for i in range(n):
        count = 1 
        while s1 and s1[-1][0] > A[i]: count += s1.pop()[1]
        left[i] = count
# coz previous elements may already popped off thus we need to maintain cnt also. 
# for example: [4,2,1] for 1, there are total of 3 num greater than it, but 4 is no longer in the stack to be counted because of the 2 alreay 
# counted 4, so the 4 was popped off before processing 1.  
        s1.append([A[i], count]) 

    # reverse the index, so we process from end to start so we can accumulate the cnt
    for j in range(n)[::-1]:
        count = 1
        while s2 and s2[-1][0] >= A[j]: count += s2.pop()[1]
        right[j] = count 
        s2.append([A[j], count])
    
    # for example [3,1,2,4], if currently processing 1, all elements that are greater than 1 on the left is [3] and all elements that are greater than 1
    # on the right are [2,4]. so total combinations that contain 1 should be cartitian product of [3, '']*[1]*['',2,4], because 1 can include 3 or not include 3, 2 options 
    # same 3 options on the right, thus 2 * 3 is the total number of combinations
    return sum(e*l*r for e,l,r in zip(A, left, right)) % (10**9 + 7) 




""" stack balance parenthsis """ 
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



# 1249 
# store the bad indices and skip them
def minRemoveToMakeValid(self, s):
    stack, ans = [], []
    bad = set()
    for i, c in enumerate(s):
        if c == '(': stack.append(i)
        
        if not stack:
            if c == ')': bad.add(i)
        else:
            if c == ')': stack.pop()   
        
    bad = set(stack) | bad
    # while stack:
    #     bad.add(stack.pop())   
    return ''.join([s[i] for i in range(len(s)) if i not in bad])
        


''' calculator problems'''
# 227 
# trick: see expression formed with negative or positive numbers. so the expression is start with positive number, thus the initial sign is '+'
# algorithm: perform on previous expression when next operator is met. because when next operator is met, you can decide the number is positive or negative.
def calculate(self, s: str) -> int:
    # get rid of spaces. 
    st = [c for c in s if c != ' ']
    n = len(st)
    # sign: last(之前一个) encountered operator. 
    stack, d, sign = [], 0, '+'
    for i in range(n):
        # handle number > 9
        if st[i].isdigit():
            d = 10 * d + int(st[i])
        
        # i == n-1, after i at the last position, where [i] must be digit, however, to avoid skipping last operator, we need to do one more operation
        if not st[i].isdigit() or i == n-1: 
            if sign == '+': 
                stack.append(d)
            elif sign == '-':
                stack.append(-d) # treat it as negative number not minus. 
            elif sign == '*':
                stack.append(stack.pop() * d)
            else:
                if stack[-1] < 0: 
                    # for python -3//2 == -2, this will cause error, thus we calculate the positve number first, then times -1
                    stack.append(-1 * (abs(stack.pop()) // d))
                else:
                    stack.append(stack.pop() // d)
            # update current sign and reset d
            d, sign = 0, st[i]
    return sum(stack)


# 224
# (: add prev res and sign to the stack
# ): add res inside cuurent matched () to whats on stack
# else: add to res one by one when next operator is met. 
def calculate(self, s: str) -> int:
    sign = '+'
    res = num = 0 
    stack = []
    for i in range(len(s)):
        if s[i].isdigit():
            num = num * 10 + int(s[i])
        elif s[i] in ['+', '-']:
            if sign == '-': num = -num
            res += num
            sign = s[i]
            num = 0
        elif s[i] == '(':
            stack.append(res)
            if sign == '+': 
                stack.append(1)
            else:
                stack.append(-1)
            res, sign = 0, '+'
        elif s[i] == ')':
            if sign == '-': num = -num
            res += num
            res = stack.pop() * res
            res += stack.pop()
            num = 0
    res += num if sign == '+' else -num
    return res



# 856
# just like recursive needing a base case, the stack is similar, when ) is encountered, 
# return either 1 or 2*cur_num
def scoreOfParentheses(self, S: str) -> int:
    stack = []
    cur_num = 0 
    for i in range(len(S)):
        if S[i] == '(':
            stack.append(cur_num)
            cur_num = 0 
        else:
            cur_num = stack.pop() + max(cur_num * 2, 1)    
    return cur_num
    

# 395
''' divide and conquor'''
# choose the rarest char every recursive call
def longestSubstring(self, s, k):
    if len(s) < k:
        return 0
    min_char = float('inf')
    char = ''
    for c in set(s):
        if min_char > s.count(c):
            min_char = s.count(c)
            char = c
    if min_char >= k:
        return len(s)
    return max(self.longestSubstring(t, k) for t in s.split(char))


# pythonic way
def longest_Substring(s, k):
    if len(s) < k: return 0 
    c = min(set(s), key=s.count)
    if s.count(c) >= k: return len(s)
    return max(longest_Substring(t, k) for t in s.split(c))


# optimize a little bit by finding the first rare character instead find the rarest character
def _longestSubstring(self, s, k):
    for c in set(s):
        if s.count(c) < k:
            return max(self._longestSubstring(t, k) for t in s.split(c))
    return len(s)



""" Heap problems """
# 23 
# tip: in python3, when the list has duplicates, you need to assign third element to be the tie breaker.
# Nlogk N is total number of nodes and K is the size of the heap. 
# worth knowing that there is PriorityQueue library in python3
def mergeKLists(self, lists: List[ListNode]) -> ListNode:
    dummy = d = ListNode(float('-inf'))
    q = []
    i = 0 
    for n in lists:
        if n: heapq.heappush(q, (n.val, i, n))
        i += 1

    while q:
        v, j, n = heapq.heappop(q)
        d.next = ListNode(v)
        d = d.next
        n = n.next
        if n: 
            i += 1
            heapq.heappush(q, (n.val, i, n))
    return dummy.next 
        


# 767 
# max heap: store (char, cnt)
def reorganizeString(self, S: str) -> str:
    freq_dic = collections.defaultdict(int)
    for c in S:
        freq_dic[c] += 1
        if freq_dic[c] > (len(S) + 1)//2: return ''

    max_heap = []
    for k, v in freq_dic.items():
        heapq.heappush(max_heap, (-1*v, k))
    res = []
    while max_heap:
        freq1, cur = heapq.heappop(max_heap)
        # if empty list or current char is not the same as the last char.
        if len(res) == 0 or res[-1] != cur:
            res.append(cur)
            freq1 += 1
            if freq1 < 0: 
                heapq.heappush(max_heap, (freq1, cur))
        else:
            freq2, nxt = heapq.heappop(max_heap)
            res.append(nxt)
            freq2 += 1
            if freq2 < 0:
                heapq.heappush(max_heap, (freq2, nxt)) 
            heapq.heappush(max_heap, (freq1, cur))# must inside the else block. there is subtle difference.
    return ''.join(res)


# trick: greedily place the most frequent char and next most frequent chars alternately. 
# use priority queue to update the most frequent and next most freuqent
def reorganizeString(self, S):
    maxheap = []
    dic = collections.defaultdict(int)
    for c in S: dic[c] += 1
    for k in dic.keys(): heapq.heappush(maxheap, (-1*dic[k], k))
    ans = []
    while len(maxheap) > 1:
        top = heapq.heappop(maxheap)
        nxt = heapq.heappop(maxheap)
        ans.append(top[1])
        ans.append(nxt[1])
        dic[top[1]] -= 1
        dic[nxt[1]] -= 1
        if dic[top[1]] > 0: 
            heapq.heappush(maxheap, (top[0]+1, top[1]))
        if dic[nxt[1]] > 0:
            heapq.heappush(maxheap, (nxt[0]+1, nxt[1]))
    
    if len(maxheap) == 1:
        f, k = heapq.heappop(maxheap)
        if -1*f > 1: return ''
        ans.append(k)
    
    return ''.join(ans)



# 621
# tip: to reduce the idle time, we need to process the most frequent tasks first because they will potentially produce the most idle time.
# algorithm: 
# put tasks frequency into max heap, process the top and if top still has some left, put them into cooldown, when cool down expired, add them back to heap again 
# to be considered. 
# when heap is empty and cooldown does not have any entry expired, meaning you need to idle the cpu.
def leastInterval(self, tasks, n):
    if n == 0: return len(tasks)
    dic = collections.defaultdict(int)
    for t in tasks: dic[t] += 1
    maxheap = [-1 * v for v in dic.values()]
    heapq.heapify(maxheap)
    cooldown = {}
    cur_time = 0
    while maxheap or cooldown:
        # 周期结束就重新加入priority queue
        if cooldown:
            if cur_time - n - 1 in cooldown:
                heapq.heappush(maxheap, cooldown[cur_time - n - 1]) 
                del(cooldown[cur_time - n - 1])
        #  insert the most frequent task
        if maxheap:
            left = heapq.heappop(maxheap) + 1
            if -1*left > 0:
                cooldown[cur_time] = left 
        # when no task in the queue or in the cooldown that is expired, the idle time will add to the cur_time.
        cur_time += 1
    return cur_time



def leastInterval(self, tasks: List[str], n: int) -> int:
    freq_counter = collections.Counter(tasks)
    max_heap = []
    for c, f in freq_counter.items():
        heapq.heappush(max_heap, (-1*f, c))
    
    cnt = 0    
    while max_heap:
        tmp, i = [], 0
        # put top n+1 items in a slot
        while i <= n:
            # if max_heap is empty but tmp has items, then meaning we are using idle to fill the rest of gaps.
            # if max_heap has items, use the items to fill the gaps. 
            cnt += 1
            if max_heap:
                freq, cur = heapq.heappop(max_heap)
                if freq +1 < 0: 
                    tmp.append((freq+1, cur))
            # if not tmp and not items in heap, meaning we are at the end nothing left to execute, 
            # no need to fill up the n cooldown time at the end. 
            if not tmp and not max_heap: break
            # no need to fill idle at the end when no items left to execute.
            i += 1
        if tmp:
            for v in tmp:
                heapq.heappush(max_heap, v)
    return cnt
            

        

# 973
# tip: heappop will not turn the list into heap automatically, it will assume current list is already a heap. 
# thus do not append item onto the list, use heappush, this will turn list into a heap
def kClosest(self, points, K):
    heap = []
    for x, y in points:
        heapq.heappush(heap, (x*x + y*y, [x, y]))
    ans = []
    while K:
        if not heap and K > 0: return []
        ans.append(heapq.heappop(heap)[1])
        K -= 1
    return ans


# 692
def topKFrequent(self, words, k):
    dic = collections.defaultdict(int)
    for w in words: dic[w] += 1
    values = [-1*i for i in dic.values()]
    h = []
    for i in zip(values, dic.keys()): 
        heapq.heappush(h, i)
    return [heapq.heappop(h)[1] for _ in range(k)]
    
        


""" Linked List """
# 234
def isPalindrome(self, head: ListNode) -> bool:
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
    # if fast, meaning from slow to fast is equal to head to slow-1
    if fast: slow = slow.next
    # no need to store prevous node, we can just traverse reverse list's length, because the previous part will have the same length
    rh = self.reverse(slow)
    while rh:
        if rh.val != head.val:
            return False
        rh = rh.next
        head = head.next 
    return True


def reverse(self, cur):
    nxt = None 
    while cur:
        tmp = cur.next
        cur.next = nxt
        nxt = cur
        cur = tmp
    return nxt


# 206
# reverse 1->None is just itself. if null list, just return null list.
def reverseList(self, head: ListNode) -> ListNode:
    if not head or not head.next: return head
    pre = head.next
    newHead = self.reverseList(pre)
    pre.next, head.next = head, None
    return newHead



# 21
# recursive 
def mergeTwoLists(self, l1: ListNode, l2: ListNode) -> ListNode:
    if not l1 or not l2: return l1 or l2
    
    if l1.val < l2.val:
        l1.next = self.mergeTwoLists(l1.next, l2)
        return l1
    else:
        l2.next = self.mergeTwoLists(l1, l2.next)
        return l2



# 426
def treeToDoublyList(self, root):
    if not root: return root 
    dummy = Node(0)
    stack = []
    d = dummy 
    while stack or root:
        while root:
            stack.append(root)
            root = root.left
            
        if stack:
            root = stack.pop()
            d.right, root.left = root, d
            d = d.right
            root = root.right
    d.right = dummy.right
    dummy.right.left = d
    return dummy.right


# 61 
# tip: connect tail with the head, find the break point cnt - k
def rotateRight(self, head, k):
    if not head or k == 0: return head 
    cur, cnt = head, 1
    while cur.next:
        cnt += 1
        cur= cur.next
    cur.next = head
    k = k % cnt
    
    dummy = ListNode(-1)
    dummy.next = head
    brk = cnt - k 
    while dummy and brk:
        dummy = dummy.next
        brk -= 1
    
    ans = dummy.next
    dummy.next = None 
    return ans
        
        
# 92
# tip: draw
def reverseBetween(self, head, m, n):
    if not head: return head
    dummy = ListNode(-1)
    dummy.next = head
    pre = dummy
    for i in range(m-1):
        pre = pre.next
    
    start = pre.next
    nxt = start.next
    
    for _ in range(n-m):
        start.next = nxt.next
        nxt.next = pre.next
        pre.next = nxt
        nxt = start.next
    
    return dummy.next
            



class Node:
    def __init__(self, val):
        self.val = val
        self.next = None


def reverseBetween(head, m, n):
    dummy = Node(-1)
    # dummy.next = head
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


# 141
# this implementation is modified version that fast and slow starts from same place
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

# this is the tortoise and hare algorithm implementation. 
# fast is one step ahead of slow, this will enhance that fast will absolutely catch up the slow.
# and it will be able to get into the while loop initially.
# this is usuful when the problem told you it has cycle in it.
def hasCycle(head):
    if not head: return False
    slow, fast = head, head.next
    
    while slow != fast:
        if not fast or not fast.next: return False
        slow = slow.next
        fast = fast.next.next
    return True


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


# tortois and hare version
def detectCycle(self, head):
    if not head: return head
    slow = head
    fast = head.next
    
    while slow != fast:
        if not fast or not fast.next: 
            return None
        slow = slow.next
        fast = fast.next.next
    # tricky part of this version is you need to increment slow by one step before running the fast and slow step by step otherwise will cause infinit loop 
    slow = slow.next
    fast = head
    while slow != fast:
        slow = slow.next
        fast = fast.next
    return fast



def plusOne_dic(head):
    if not head: return 0
    dummy = Node(-1)
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
    dummy = Node(0)
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

# 445
# tip: use stack 
def addTwoNumbers(self, l1, l2):
    stack1, stack2 = [], []
    
    while l1: 
        stack1.append(l1)
        l1 = l1.next
    
    while l2:
        stack2.append(l2)
        l2 = l2.next
    
    head, carry = None, 0
    while stack1 or stack2 or carry:
        a = stack1.pop().val if stack1 else 0
        b = stack2.pop().val if stack2 else 0 
        carry, d = divmod(a + b + carry, 10)
        node = ListNode(d)
        node.next = head
        head = node
    return head


def mergeTwoLists_iter(l1, l2):
    dummy = Node(-1)
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
    d1, d2 = Node(-1), Node(-1)
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

#   intersect at a node or at null
#   should include the null node to end the while condition
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

# 19
def removeNthFromEnd(head, n):
    if not head: return head
    p1 = p2 = head
    pre = None
    while p1:   
        while n:
            p1 = p1.next
            n -= 1
        # when n is the length of the linkedlist, that means remove head
        if not p1:
            return p2.next
        pre = p2 
        p1 = p1.next
        p2 = p2.next
    pre.next = p2.next
    p2.next = None
    return head

# 143
# trick: use the mid point, so only need to reverse the second part.
def reorderList(self, head):
    if not head: return []
    h = head
    second_head, prev = self.find_mid(h)
    prev.next = None
    new_head = self.reverse(second_head)
    self.merge(h, new_head)
    return head


def find_mid(self, h):
    slow, fast = h, h 
    pre = None
    while fast and fast.next:
        print(fast.val)
        pre = slow
        slow = slow.next
        fast = fast.next.next
    # make sure second part will alwasys less or equal to first part in length 
    # if two mid nodes, pick the second one
    if fast: return slow.next, slow
    return slow, pre


def merge(self, h1, h2):
    while h1 or h2:
        tmp1 = h1.next 
        h1.next = h2 
        if h2: 
            tmp2 = h2.next
            h2.next = tmp1
            h2 = tmp2
        h1 = tmp1
        
    
def reverse(self, cur):
    h = None 
    while cur: 
        tmp = cur.next
        cur.next = h
        h = cur
        cur = tmp 
    return h


def mergeKLists(lists):
    if not lists: return None
    def merge(h1, h2):
        d = cur = Node(-1)
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
    dummy = pre = Node(-1)
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
        dummy = Node(-1)
        d = dummy
        while h1 and h2:
            if h1.val < h2.val:
                dummy.next = h1
                h1 = h1.next
            else:
                dummy.next = h2
                h2 = h2.next
            dummy = dummy.next
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
    d = Node(-1)
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


# 25
def reverseKGroup(head, k):
    def reverse(h1, h2):
        next = None
        while h1 != h2:
            tmp = h1
            h1 = h1.next
            tmp.next = next
            next = tmp
        return next
    dummy = Node(-1)
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


""" Dynamic programming and Greedy """
# greedy
#406
def reconstructQueue(self, people: List[List[int]]) -> List[List[int]]:
    dic = {}
    for h, k in people:
        if h in dic:
            dic[h].append(k)
        else:
            dic[h] = [k]
    res = []
    for h in sorted(dic.keys())[::-1]:
        same_hight = sorted(dic[h])
        for i in same_hight:
            res = res[:i] + [[h, i]] + res[i:]
    return res


# 55
# as long as j can reach i, where i can reach the top, then j can reach top 
# convert the probelm to: find the first position that can reach the goal, check if that position is position 0. 
def canJump_greedy(self, nums):
    goal = len(nums)-1
    for i in range(len(nums)-1, -1, -1):
        if i + nums[i] >= goal:
            goal = i 
    return goal == 0


# 1092
def shortestCommonSupersequence(self, str1, str2):
    lcs = self.LCS(str1, str2)
    print(lcs)
    res, i, j = [], 0, 0
    for c in lcs: 
        while str1[i] != c:
            res.append(str1[i])
            i += 1
        
        while str2[j] != c:
            res.append(str2[j])
            j += 1
        res.append(c)
        i, j = i+1, j+1
        
    return ''.join(res) + str1[i:] + str2[j:]
            
            
def LCS(self, s1, s2): 
    n, m = len(s1), len(s2)
    dp = [['' for _ in range(n+1)] for _ in range(m+1)]
    
    for i in range(1, m+1):
        for j in range(1, n+1):
            if s2[i-1] == s1[j-1]:
                dp[i][j] = dp[i-1][j-1] + s1[j-1]
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1], key=len)
    
    return dp[-1][-1]


# recursive relationship not necessarily between current and previous one, it could be some gap between current and previous.
# 1024
def videoStitching(self, clips, T):
    dp = [T+1] * (T+1)
    dp[0] = 0 
    for i in range(1, T+1):
        for c in clips:
            if c[0] <= i <= c[1]:
                dp[i] = min(dp[i], dp[c[0]] + 1)
    return dp[-1] if dp[-1] != T+1 else -1
    


# 1143
def longestCommonSubsequence(self, text1, text2):
        dp = [[0 for _ in range(len(text1)+1)] for _ in range(len(text2)+1)]
        
        for i in range(1, len(text2)+1):
            for j in range(1, len(text1)+1): 
                if text2[i-1] == text1[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
        return dp[-1][-1]


""" optimal space solution """


# 70
# memo
def climbStairs(self, n: int) -> int:
    return self.climb(n, {})
    

def climb(self, n, dic):
    if n < 0: return 0
    if n == 0: return 1 
    if n in dic: return dic[n]
    ways = self.climb(n-1, dic) + self.climb(n-2,dic)
    dic[n] = ways
    return ways


# DP
def climbStairs(self, n: int) -> int:
    dp = [0]*(n+1)
    dp[0] = 1
    for i in range(1, n+1): 
        dp[i] += dp[i-1] if i-1 >= 0 else 0 
        dp[i] += dp[i-2] if i-2 >= 0 else 0
    return dp[n]
    

# space optimized
def climbStairs(self, n: int) -> int:
    ways, prepre, pre = 0, 0, 1
    for _ in range(1, n+1):
        ways = pre + prepre
        prepre = pre
        pre = ways
    return ways


# more space optimized
def climbStairs(self, n: int) -> int:
    ways, pre = 1, 0
    for _ in range(1, n+1):
        tmp = ways
        ways += pre
        pre = tmp
    return ways


# 746
# memo
def minCostClimbingStairs(self, cost: List[int]) -> int:
    return self.minCost(cost, len(cost), {})

# cheapest cost to go to the top
# cost to get to step i + cost to leave step i to get to top
def minCost(self, cost, top, dic):
    if top == 0 or top == 1: return 0 
    if top in dic: return dic[top]
    dic[top] = min(self.minCost(cost, top-1, dic)+cost[top-1], self.minCost(cost, top-2, dic) + cost[top-2])
    return dic[top]
        


# DP
# dp[i]: min cost to get to step i before leaving step i.
def minCostClimbingStairs(self, cost: List[int]) -> int:
    dp = [0] * (len(cost)+1)
    for i in range(2, len(cost)+1):
        dp[i] = min(dp[i-1]+cost[i-1], dp[i-2]+cost[i-2])
    return dp[-1] # we dont leave destination



# 303
class NumArray(object):

    def __init__(self, nums):
        """
        :type nums: List[int]
        """
        self.nums = nums    
        self.prefix= [0]*(len(nums)+1)

        for i in range(1, len(self.prefix)):
            self.prefix[i] = self.prefix[i-1] + nums[i-1]
        
        
    def sumRange(self, i, j):
        return self.prefix[j+1] - self.prefix[i]


""" max sum/product subarray """
# 53
# it asks for subarray, means you have to include every elements in some range [i, j], not like knapsack problem you can omit some elements. 
def maxSubArray(nums):
    local_max = global_max = nums[0]
    for i in range(1, len(nums)):
        local_max = max(nums[i], local_max+nums[i])
        # because dp[-1] is not the answer, it's just the max subarray ending at i, thus you need to find the max(dp)
        global_max = max(global_max, local_max)
    return global_max


# similar to above but it doesnt maintain local, it just overwrite in the array. 
def maxSubArray_Kadane(nums):
    max_sum = nums[0]
    for i in range(1, len(nums)):
        if nums[i-1] > 0:
            nums[i] += nums[i-1]
        max_sum = max(max_sum, nums[i])
    return max_sum


# divideAndConquer
# the solution is among three location, either left side and right side or in the middle
def maxSubArray_divideAndConquer(nums):
    # left and right is the boundres of subarray
    def cross_sum(nums, mid, left, right):
        if left == right:
            return nums[left]
        left_max_sum = float('-inf')
        left_cur_sum = 0
        for i in range(left, mid)[::-1]:
            left_cur_sum += nums[i]
            left_max_sum = max(left_max_sum, left_cur_sum)

        right_max_sum = float('-inf')
        right_cur_sum = 0
        for j in range(mid, right+1):
            right_cur_sum += nums[j]
            right_max_sum = max(right_max_sum, right_cur_sum)
        return left_max_sum + right_max_sum

    def merge_sum(nums, left, right):
        # base case
        if left == right:
            return nums[left]
        mid = (left+right)//2
        left_sum = merge_sum(nums, left, mid)
        right_sum = merge_sum(nums, mid+1, right)
        cross_sm = cross_sum(nums, mid, left, right)
        return max(left_sum, right_sum, cross_sm)
    if not nums:
        return 0
    return merge_sum(nums, 0, len(nums)-1)




# 152
# min_dp[i]: minimum product for subarray ending with [i]
# max_dp[i]: maximum product for subarray ending with [i]
# res[i]: largest product prefix i elements
# O(2*n) spcace
def maxProduct(self, nums):
    dp_max = [1] * (len(nums)+1)
    dp_min = [1] * (len(nums)+1)  
    res = float('-inf')
    for i in range(1, len(nums)+1):
        dp_max[i] = max(dp_min[i-1] * nums[i-1], dp_max[i-1] * nums[i-1], nums[i-1])
        dp_min[i] = min(dp_min[i-1] * nums[i-1], dp_max[i-1] * nums[i-1], nums[i-1])
        res = max(res, dp_max[i])  
    return res 
            

# space O(1)

def maxProduct_optimal(nums):
    n = len(nums)
    min_prev = nums[0]
    max_prev = nums[0]
    res = nums[0]
    for i in range(1, n):
        if nums[i] > 0:
            min_prev = min(min_prev*nums[i], nums[i])
            max_prev = max(max_prev*nums[i], nums[i])
        else:
            tmp = min_prev
            min_prev = min(max_prev*nums[i], nums[i])
            max_prev = max(tmp*nums[i], nums[i])

        res = max(res, max_prev)
    return res


"""buy/sell stock problems """
# p121
# same idea as maxsubarry problem.
# when buy at [i] sell at [j], no matter how the price fluctuated, the profit p[j]-p[i] = every profit/loss(loss that does not exceed the previous profit generated) generated in between subarry[i : j]
# therefore we turn this problen into looking for a subarray such that has the max sum of profit


def maxProfit(prices):
    max_cur_profit = max_profit = 0
    for i in range(1, len(prices)):
        # max_cur_profit + (prices[i] - prices[i-1]) > 0, we will keep the profit else we just ignor previous section and
        # start over again and max_cur_profit set to 0
        #prices[i] - prices[i-1]) can be optimize to 0, becuase the problem says we can make at most 1 buy and sell.
        # when there is a loss if we sell, we can not doing trasaction, so the profit is 0 at most.
        max_cur_profit = max(max_cur_profit + (prices[i] - prices[i-1]), 0)
        max_profix = max(max_profit, max_cur_profit)
    return max_profit


# profit will exist in some range [i:j] range sum >= 0 
def maxProfit(self, prices):
    max_prof = cur = 0
    for i in range(1, len(prices)):
        diff = prices[i] - prices[i-1]
        if cur + diff >= 0:
            cur += diff
            max_prof = max(cur, max_prof)
        else:
            # if current accumulative sum is not able to compensate current loss, we should not include the loss, we should 
            # reset the cur to 0 to find next range that will generate profit. 
            cur = 0
    return max_prof


# idea: find the smallest price point find the first peak following the point
def maxProfit(self, prices):
    minprice = float('inf')
    prof = 0
    for i in range(len(prices)):
        minprice = min(minprice, prices[i])
        # if prices[i] > minprice:
        prof = max(prof, prices[i] - minprice)
    return prof    
    

# conclusion:
# for maxprofit probelems, there is a hidden factor which is vital to solve the probelems:
# number of stock in hand.
# when number of stock in hand is 0, then you are allow to buy or rest.
# when number of stock in hand is 1, then you are allow to sell or rest.



# p309
# define dp:
# buy[i]: max profit until i and the last action done is buy but not necessarilly at i position
# sell[i]: max profit unitl i and the last action done is sell but not necessarilly at i position
# note that: the n-1 position(last position), we can possiblly doing sell at n-1, but not buy, the profit sell[n-1] is the max profit.

# define recursive relation:
# for each position i, you can buy or not buy and sell or not sell.
# to determine whether to buy or not buy at ith day:
# to not buy, if we bought already before i-1 or on i-1, at i we can take rest.
# to buy, if we sell at i-2, take a rest at i-1, then we can buy at i.
# buy[i] = max(buy[i-1], sell[i-2] - prices[i])

# to determine whether to sell or not sell at ith day:
# to not sell: we can sell at i-1 and rest at i
# to sell: we can buy at i-1 and sell at i
# sell[i] = max(sell[i-1], buy[i-1]+prices[i])

# optimization: constant space
# we can see that the solution is only depending on i, i-1, i-2
# so we can use variables: b0, b1  = buy[i], buy[i-1] and s0, s1, s2  = sell[i], sell[i-1], sell[i-2]
# recursive relation becomes: b0 = max(b1, s2 - prices[i]) and s0 = max(s1, b1+prices[i])
# then update the b0, b1 and s0, s1, s2 correspondingly.

# initialization:
# you cannot sell at beginning without buying, so s0 = s1 = s2 = 0 and b0 = b1 = -prices[0]
def maxProfit_cooldown(prices):
    if not prices or len(prices) <= 1:
        return 0
    b0 = b1 = -prices[0]
    s0 = s1 = s2 = 0
    for i in range(1, len(prices)):
        b0 = max(b1, s2 - prices[i])
        s0 = max(s1, b1 + prices[i])
        b1, s1, s2 = b0, s0, s1
    return s0


""" state machine thought process """
# for every iteration, you will end up in three states and maximize them: [ready to buy], [ready to sell], [sold and cool down]. 
def maxProfit_cool(self, prices):
    prev_to_buy = 0
    prev_to_sell = cool = float('-inf')
    
    for p in prices:
        # for current state to be no stock and ready to buy, either we are previously ready to buy but buy at current p or 
        # previously in cooling state and not buying at current state. 
        cur_to_buy = max(prev_to_buy, cool)
        # for current state to be have a stock and ready to sell, either previously already ready for sell but still not selling at current p 
        # or previously ready to buy and buy it today at p
        cur_to_sell = max(prev_to_sell, prev_to_buy - p)
        # previously ready to sell and sell it at p today. 
        cool = prev_to_sell + p 
        prev_to_buy, prev_to_sell = cur_to_buy, cur_to_sell
    return max(prev_to_buy, cool)


# p714
# similar to above solution
def maxProfit_basic(self, prices, fee):
    prev_to_buy, prev_to_sell = 0, float('-inf')  
    for p in prices: 
        cur_to_buy = max(prev_to_buy, prev_to_sell + p - fee)
        cur_to_sell = max(prev_to_sell, prev_to_buy - p)
        prev_to_buy, prev_to_sell = cur_to_buy,  cur_to_sell
    return max(prev_to_buy, prev_to_sell)


def maxProfit_ii(self, prices, fee):
    buy, sell = 0, float('-inf')  
    for p in prices: 
        buy = max(buy, sell + p - fee)
        sell = max(sell, buy - p)
    return max(buy, sell)


# p123
def maxProfit_twoTransactions(prices):
    release_first = 0
    release_second = 0
    hold_first = float('-inf')
    hold_second = float('-inf')
    for price in prices:
        # 1 in hand at first transaction
        hold_first = max(hold_first, -price)
        # 0 in hand at first transaction
        release_first = max(release_first, hold_first+price)
        # 1 in hand at second transaction
        hold_second = max(hold_second, release_first-price)
        # 0 in hand at second transaction
        release_second = max(release_second, hold_second+price)
    return release_second



# p188
# key: only increment k when buy stock not sell.
def maxProfit_k(k, prices):
    # when k >= len/2, this is same question asking for unlimited number of transactions
    if k >= len(prices)//2:
        profit = 0
        for i in range(1, len(prices)):
            if prices[i] > prices[i-1]:
                profit += prices[i] - prices[i-1]
        return profit

    ready_to_sell = [float('-inf')]*(k+1)
    ready_to_buy = [0]*(k+1)

    for price in prices:
        # each transaction has two states
        for i in range(1, k+1):
            # assume buy first
            ready_to_sell[i] = max(ready_to_sell[i], ready_to_buy[i-1] - price)
            ready_to_buy[i] = max(ready_to_buy[i], ready_to_sell[i] + price)
    return ready_to_buy[k]



""" house robber problems """
# p198
# recursive top-down memo AC 
def rob(self, nums):
        if not nums: return 0 
        memo = {}
        return self.rob_helper(nums, len(nums)-1, memo)


def rob_helper(self, nums, i, memo):
    if i in memo: 
        return memo[i]
    if i < 0:
        return 0 
    memo[i-2] = self.rob_helper(nums, i-2, memo)
    return max(memo[i-2] + nums[i], self.rob_helper(nums, i-1, memo)) 


# O(n) space bottom up 
# dp[i]: max amount prefix i. 
# dp[i] = max(dp[i-1], dp[i-2] + [i-1])
def rob(nums):
    n = len(nums)
    dp = [0]*(n+1)
    dp[0] = 0
    for i in range(1, n+1):
        dp[i] = max(dp[i-1], dp[i-2]+nums[i-1] if i-2 >= 0 else nums[i-1])
    return dp[n]


# just like fibnacci sequence, you can use two variable to replace the array
# because the solution only depends on previous two value, so we can optimize the solution Use variable only
def rob_optimalspace(nums):
    cur, pre = 0, 0
    for num in nums:
        tmp = cur
        cur = max(cur, pre+num)
        pre = tmp
    return cur



# p213
# hint: circle actually split the problem into two subproblems and we can solve it using above code
# only two scenarios, max money to rob in nums[0: len(nums)-1] and [1:len(nums)]
def rob_circle(nums):
    # same as above code
    def helper(nums):
        n = len(nums)
        cur, pre = 0, 0
        for i, num in enumerate(nums):
            tmp = cur
            cur = max(cur, pre+num)
            pre = tmp
        return cur
    if not nums:
        return 0
    if len(nums) == 1:
        return nums[0]
    return max(helper(nums[:-1]), helper(nums[1:]))



# p740
# reduce this problem to rober house
# so we can group all the same points into a house
# then we just follow the rober house rule to rob each house
def deleteAndEarn(nums):
    if not nums:
        return 0
    # dp[i] is prefix i houses to get the max profit and you cannot rob consecutive houses. 
    def rob(g):
        dp = [0]*(len(g)+1)
        dp[0] = 0
        dp[1] = g[0]
        for i in range(2, len(g)+1):
            dp[i] = max(dp[i-1], dp[i-2] + g[i-1])
        return dp[len(g)]

    rng = max(nums)
    groups = [0]*(rng+1)
    for num in nums:
        groups[num] += num
    return rob(groups)


# 139
# dfs memo 
def wordBreak(self, s: str, wordDict: List[str]) -> bool:
    return self.can_segament(0, s, set(wordDict), set())
        
def can_segament(self, cur, s, wordset, dic):
    if cur == len(s): return True
    if cur in dic: return False
    for i in range(cur, len(s)):
        w = s[cur: i+1]
        if w in wordset and self.can_segament(i+1, s, wordset, dic):
            return True
    dic.add(cur)
    return False


# dp
# dp[i]: check if prefix i characters/substring end with [i],  can be segmented into words in given list
def wordBreak_segment(s, wordDict):
    dp = [False]*(len(s)+1)
    dp[0] = True
    for i in range(1, len(s)+1):
        for j in range(i):
            if dp[j] and s[j:i] in wordDict:
                dp[i] = True
                break
    return dp[len(s)]


# bfs 
def wordBreak_bfs(self, s, wordDict):
    wd, visited = set(wordDict), set()
    queue = collections.deque()
    queue.append(0)
    while queue:
        pre = queue.popleft()
        for i in range(pre+1, len(s)+1):
            # segment at [i: ] will not return true, dont waste your time, move on.
            if i in visited: continue
            if s[pre: i] in wd:
                if i == len(s): return True
                queue.append(i)
                visited.add(i)
    return False



# p140
# dfs + dp
def wordBreak_addSpace(s, wordDict):
    def dfs(s, dic, path, res):
        if wordBreak_segment(s, dic):
            if not s:
                res.append(path.strip())
                return 

            for i in range(1, len(s)+1):
                if s[:i] in dic:
                    dfs(s[i:], dic, path + s[:i] + ' ', res)
    

    def wordBreak_segment(s, wordDict):
        dp = [False]*(len(s)+1)
        dp[0] = True
        for i in range(1, len(s)+1):    
            for j in range(i):
                if dp[j] and s[j:i] in wordDict:
                    dp[i] = True
                    break
        return dp[len(s)]

    res = []
    wordDict = set(wordDict)
    dfs(s, wordDict, '', res)
    return res



# p120
# ajacent here means below and right. 
# the min path must end with one of the last row's element, so we do a bottom up search for each elements at th bottom.
def minimumTotal(triangle):
    dp = [[triangle[0][0]
           for _ in range(len(triangle[i]))] for i in range(len(triangle))]
    for i in range(1, len(triangle)):
        for j in range(len(triangle[i])):
            above = dp[i-1][j] if j < len(triangle[i-1]) else float('inf')
            above_left = dp[i-1][j-1] if j-1 >= 0 else float('inf')
            dp[i][j] = min(above, above_left) + triangle[i][j]
    return min(dp[len(triangle)-1])




# p64
# top down memo time O(n*m), space O(n*m)
def minPathSum(self, grid):
    if not grid: return 0
    return self.min_path(grid, len(grid)-1, len(grid[0])-1, {})
    
#min path from top left to bottom right
def min_path(self, grid, i, j, dic):
    if i == 0 and j == 0: return grid[0][0]
    if (i, j) in dic: return dic[(i,j)]
    left = self.min_path(grid, i, j-1, dic) if j - 1 >= 0 else float('inf')
    top = self.min_path(grid, i-1, j, dic) if i - 1 >= 0 else float('inf')
    ans = min(left, top) + grid[i][j]  
    dic[(i, j)] = ans
    return ans 

   
# 2d dp 
def minPathSum(self, grid):
    n, m = len(grid), len(grid[0])
    dp = [[0 for _ in range(m)] for _ in range(n)]
    dp[0][0] = grid[0][0]
    for j in range(1, m):
        dp[0][j] = dp[0][j-1] + grid[0][j]
    
    for i in range(1,n):
        dp[i][0] = dp[i-1][0] + grid[i][0]
    
    for i in range(1, n):
        for j in range(1, m):
            dp[i][j] = min(dp[i-1][j], dp[i][j-1]) + grid[i][j]
    
    return dp[n-1][m-1]


# 1d dp
# only relate to row/col before, we can resue the col before once we calculated the result and assign back to the col.
def uniquePaths(self, m, n):
    dp = [0] * m
    dp[0] = 1
    for _ in range(n):
        for j in range(1,m):
            dp[j] = dp[j] + dp[j-1]
    return dp[m-1]



# p62
# top-down-memo
def uniquePaths_memo(self, m, n):
    return self.dfs(m-1, n-1, {})


def dfs(self, i, j, dic):
    if i == 0 and j == 0: return 1 
    if (i, j) in dic: return dic[(i, j)]
    top = self.dfs(i-1, j, dic) if i-1>=0 else 0
    left = self.dfs(i, j-1,dic) if j-1>=0 else 0 
    dic[(i, j)] = top + left
    return top + left


# two d
def uniquePaths_2d(self, m, n):
    dp = [[1 for _ in range(m)] for _ in range(n)]
    for i in range(1, n):
        for j in range(1, m):
            dp[i][j] = dp[i-1][j] + dp[i][j-1]
    return dp[-1][-1]

# 1d
def uniquePaths(self, m: int, n: int) -> int:
    dp = [1] * m
    for i in range(1, n):
        for j in range(1, m):
            dp[j] += dp[j-1] if j-1 >= 0 else 0
    return dp[m-1]
            


# p63
# we can only go right and down. 
# for the first row and column, if there is a positon i is 1, the position i and after will no longer reachable.
def uniquePathsWithObstacles(self, grid):
    n, m = len(grid), len(grid[0])
    dp = [[1 for _ in range(m)] for _ in range(n)]      
    
    for i in range(n):
        if grid[i][0] == 1: 
            for j in range(i, n):
                dp[j][0] = 0

    for i in range(m):
        if grid[0][i] == 1: 
            for j in range(i, m):
                dp[0][j] = 0
    
    for i in range(1,n):
        for j in range(1, m):
            dp[i][j] = (dp[i-1][j] + dp[i][j-1]) if grid[i][j] != 1 else 0    
            
    return dp[-1][-1]



# better implementation using prefix i, j. this will avoid edge cases handling like above. 
def uniquePathsWithObstacles(obstacleGrid):
    m, n = len(obstacleGrid), len(obstacleGrid[0])
    dp = [[0 for _ in range(n+1)] for _ in range(m+1)]
    for i in range(1, m+1):
        for j in range(1, n+1):
            if obstacleGrid[i-1][j-1] != 1:
                if i == 1 and j == 1:
                    dp[i][j] = 1
                else:
                    dp[i][j] = dp[i-1][j] + dp[i][j-1]
    return dp[m][n]




# p931
# inplace 
def minFallingPathSum(self, A):
    if not A: return 0
    n,m = len(A), len(A[0])
    for i in range(n):
        for j in range(m):
            if i > 0: 
                above = A[i-1][j]
                left = A[i-1][j-1] if j-1 >= 0 else float('inf')
                right = A[i-1][j+1] if j+1 < m else float('inf')
                A[i][j] += min(above, left, right)
    return min(A[-1])



# 279
# top-down memo
def numSquares(self, n: int) -> int:
    lst = [i*i for i in range(1, int(n**(0.5))+1)]
    return self.min_cnt(n, lst, {})



def min_cnt(self, n, lst, dic):
    if n in lst: return 1
    if n in dic: return dic[n]
    
    mi = float('inf')
    for k in lst:
        if n - k < 0: break
        tmp = self.min_cnt(n-k, lst, dic)+1
        mi = min(mi, tmp)
    dic[n] = mi
    return mi


# intuition: problem asks for minimum number of square count, thus there are choices to make to abtain the optimal solution
# so we use dp to sovle this problem. a given number subtracts number i square where i*i <= given number. compare all the possible
# i square number being subtracted from given number, we can remainder R, compare all dp[R] find the minimum
# so we can write the recurive relation as, dp[n] = min(dp[n-i*i],...) + 1, i*i <= n and i>= 1
# plus 1 in above becuase we subtracted a square number out, so we need to add that number's count back, which is 1.

def numSquares(n):
    dp = [0]*(n+1)
    for i in range(1, n+1):
        j = 1
        remainder = i
        while i >= j*j:
            remainder = min(remainder, dp[i-j*j])
            j += 1
        dp[i] = remainder + 1
    return dp[n]


# 72
# top down memo 
def minDistance(self, word1: str, word2: str) -> int:
    return self.findMinEdit(0, 0, word1, word2, {})
    
    
def findMinEdit(self, i, j, w1 ,w2, dic):
    if i == len(w1) or j == len(w2):
        return (len(w2) - j) or (len(w1) - i)
    
    if(i, j) in dic: return dic[(i,j)]
    
    cnt = float('inf')
    if w1[i] == w2[j]:
        cnt = self.findMinEdit(i+1, j+1, w1, w2, dic)
    else:
        # replace
        cnt = min(cnt, self.findMinEdit(i+1, j+1, w1, w2, dic) + 1)
        # delete
        cnt = min(cnt, self.findMinEdit(i+1, j, w1, w2, dic) + 1)
        # insert
        cnt = min(cnt, self.findMinEdit(i, j+1, w1, w2, dic) + 1)
    dic[(i, j)] = cnt
    return cnt


# bottom up dp
def minDistance(self, word1, word2):
    n, m = len(word1), len(word2)
    dp = [[0 for _ in range(m+1)] for _ in range(n+1)]
    
    for i in range(n+1):
        dp[i][0] = i
    
    for j in range(m+1):
        dp[0][j] = j
        
    for i in range(1,n+1):
        for j in range(1,m+1):
            if word1[i-1] == word2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                # insertion, deletion, replacement.
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    
    return dp[-1][-1]
                    

# p338
# O(n)


def countBits(num):
    dp = [0]*(num+1)
    dp[0] = 0
    for i in range(1, num+1):
        if i % 2 == 0:
            dp[i] = dp[i//2]
        else:
            dp[i] = dp[i-1]+1
    return dp


# p312
# intuition: when there are many choices to make and a choice you make will leave you with many subproblems, which are very similar, to solve
# this indicates that we should use dynamic programming or memoization or divide and conquar
# if we try to find which one to burst first, then each choice will create different subproblems because the this ajacentcy changes
# so we need to think reversely, how to avoid changing ajacency. we can find which balloon to burst last and the previous bursted balloons
# will not affect the last balloon's ajacency, so we only need to iterate each balloon as the last to burst.
# recursively find the max coins collected for the parts on the both sides of k

# dp[i, j]: the max coie can collect in range of i and j, exclusive
# dp[i, j] = dp[i+1, k-1] + dp[k+1, j] + num[k]*nums[i]*nums[j]
# dp[0, n] is the result
def maxCoins(nums):
    new_nums = [1] + nums + [1]
    n = len(new_nums)
    dp = [[0 for _ in range(n)] for _ in range(n)]

    def burst(nums, dp, left, right):
        if right - left <= 1:
            return 0  # no balloons in between return 0
        if dp[left][right] > 0:
            return dp[left][right]  # cached before
        ans = 0
        for k in range(left+1, right):
            # burst k balloon last
            ans = max(ans, burst(nums, dp, left, k) + burst(nums, dp, k, right) + nums[left]*nums[k]*nums[right])
        dp[left][right] = ans
        return dp[left][right]
    return burst(new_nums, dp, 0, n-1)


# p32
# dp[i]: longest valid substring ending with ith parenthesis
# we know that valid substring must end with closing parenthesis ')', if ith parenthesis is '(', then its dp value would be 0
# if s[i] == ')' and s[i-1] == '(', dp[i] = dp[i-2]+2
# if s[i] == s[i-1] == ')' and s[i-dp[i-1]-1] == '(', then dp[i] = dp[i-1]+2 + dp[i-dp[i-1]-2]
def longestValidParentheses(s):
    n = len(s)
    dp = [0]*n
    for i in range(1, n):
        if s[i] == ')':
            if s[i-1] == '(':
                dp[i] = 2 + (dp[i-2] if i-2 >= 0 else 0)
            elif s[i-1] == ')':
                if i-dp[i-1]-1 >= 0 and s[i-dp[i-1]-1] == '(':
                    dp[i] = dp[i-1] + 2 + (dp[i-dp[i-1]-2] if i-dp[i-1]-2 >= 0 else 0)
    return max(dp)



# p91
# similar to climb steps
# dp[i]: number of ways to decode prefix i substring.
# if [i:i+1] is in range [1, 9], then dp[i] = dp[i] + dp[i-1]
# if [i-1: i+1] is in range [10, 26], then dp[i] = dp[i] += dp[i-2]
# dp[i] is the sum of dp[i-1] and dp[i-2] if one and two digits are in range
def numDecodings(s):
    if not s:return 0
    n = len(s)
    dp = [0]*(n+1)
    dp[0] = 1
    dp[1] = 1 if s[0] != '0' else 0
    for i in range(2, n+1):
        if 1 <= int(s[i-1:i]) <= 9:
            dp[i] += dp[i-1]
        if 10 <= int(s[i-2: i]) <= 26:
            dp[i] += dp[i-2]
    return dp[n]




# p115
# dp[i][j] number of subsequences s[:i] will equal to t[:j]
# dp[*][0] = 1
def numDistinct(s, t):
    n = len(s)
    m = len(t)
    if n < m:
        return 0
    dp = [[0 for _ in range(m+1)] for _ in range(n+1)]
    for i in range(n+1):
        dp[i][0] = 1

    for i in range(1, n+1):
        for j in range(1, m+1):
            # when curent char is equal, so current char either use or not use, so we sum them up to get the total subsequences
            if s[i-1] == t[j-1]:
                dp[i][j] = dp[i-1][j-1] + dp[i-1][j]
            else:
                dp[i][j] = dp[i-1][j]
    return dp[n][m]



# 221
# dp[i][j] the maximum side of the square right corner ended at [i-1, j-1], for prefix i and j 
# dp[i][j] = min(dp[i][j-1], dp[i-1][j], dp[i-1][j-1])+1
# the maximum side of squre ended at [i][j] is bounded by the min side of top, left, and top left square side size.
# be careful with python circular array feature, can be pitfall or implicit bug.
def maximalSquare(matrix):
    if not matrix:
        return 0
    n = len(matrix) 
    m = len(matrix[0])
    dp = [[0 for _ in range(m+1)] for _ in range(n+1)]
    max_side = 0
    for i in range(1, n+1):
        for j in range(1, m+1):
            if matrix[i-1][j-1] == '1':
                dp[i][j] = min(dp[i][j-1], dp[i-1][j], dp[i-1][j-1])+1
                max_side = max(max_side, dp[i][j])
    return max_side * max_side


# top down memo
def maximalSquare(self, matrix):
    def max_sqr(matrix, x, y, dic):
        if x >= len(matrix) or y >= len(matrix[0]) or matrix[x][y] == '0': return 0
        if (x, y) in dic and dic[(x, y)] > 0: return dic[x, y]
        res = min(max_sqr(matrix, x+1, y, dic), max_sqr(matrix, x, y+1, dic), max_sqr(matrix, x+1, y+1, dic))+1
        dic[(x, y)] = res
        return res
    
    max_side, dic = 0, {}
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            max_side = max(max_side, max_sqr(matrix, i, j, dic))
    return max_side * max_side





# p413
# dp[i] is number of slices ending with ith element
# a bit tricky to recognize that all dp[i-1]'s slices ending with i-1 th element adding ith element will also form valid slices if the [i] - [i-1] == [i-1] - [i-2] is passed.
# on the other hand, [i], [i-1], [i-2] forms additional one slice
# thus, dp[i] = dp[i-1]+1
def numberOfArithmeticSlices(A):
    n = len(A)
    dp = [0]*n
    res = 0
    for i in range(2, n):
        if A[i] - A[i-1] == A[i-1] - A[i-2]:
            # when you add one to the sequence, it will form an extra minimal qualified length of slice and if it pass the above condition, then add 1.
            dp[i] = dp[i-1] + 1
            res += dp[i]
    return res



# p446
# dp[i][j]: number of as subseq ending i with diff j 
# dp[i][j] = sum(dp[k][j]) + 1 while 0<= k < i 
# dp[0][j] = 0
def numberOfArithmeticSlices_ii(A):
    pass



# 97
# dp[i][j]: whether prefix i and prefix j of s1, s2 can form interleaving string s3[i+j-1]
# dp[i][j] = (dp[i][j-1] and s2[j-1] == s3[i+j-1]) or (dp[i-1][j] and s[i-1] == s3[i+j-1])
def isInterleave(s1, s2, s3):
    n, m, k = len(s1), len(s2), len(s3)
    if (n+m) != k: return False
    dp = [[False for _ in range(m+1)] for _ in range(n+1)]
    for i in range(n+1):
        for j in range(m+1):
            if i == 0 and j == 0:
                dp[i][j] = True
            elif i == 0:
                dp[i][j] = dp[i][j-1] and s2[j-1] == s3[j-1]
            elif j == 0:
                dp[i][j] = dp[i-1][j] and s1[i-1] == s3[i-1]
            else:
                dp[i][j] = (dp[i][j-1] and s2[j-1] == s3[i+j-1]) or (dp[i-1][j] and s1[i-1] == s3[i+j-1])
    return dp[n][m]




# p72
def minDistance(word1, word2):
    n = len(word1)
    m = len(word2)
    dp = [[0 for _ in range(m+1)] for _ in range(n+1)]
    for i in range(n+1):
        dp[i][0] = i
    for j in range(m+1):
        dp[0][j] = j

    for i in range(1, n+1):
        for j in range(1, m+1):
            if word1[i-1] != word2[j-1]:
                dp[i][j] = min(dp[i-1][j-1], dp[i-1][j], dp[i][j-1]) + 1
            else:
                dp[i][j] = dp[i-1][j-1]
    return dp[n][m]


# p741
# this problem is tricky if you do twice dp, one downward on upward, will not get correct answer. for this example:
# 11100
# 00101
# 10100
# 00100
# 00111
# answer will off by one because two dp is some kind of greedy, this is not right

def cherryPickup(grid):
    pass

    ################################ wrong answer using two dp
    # res = 0
    # n = len(grid)
    # fw = [[0 for _ in range(n+1)] for _ in range(n+1)]
    # bw = [[0 for _ in range(n+1)] for _ in range(n+1)]
    # for i in range(1, n+1):
    #     for j in range(1, n+1):
    #         if grid[i-1][j-1] == -1:
    #             fw[i][j] = float('-inf')
    #             continue
    #         fw[i][j] = max(fw[i-1][j], fw[i][j-1]) + (1 if grid[i-1][j-1] == 1 else 0)

    # i = j = n
    # fw[i][j] = 0
    # while i >= 1 and j >= 1 and fw[i][j] != 0:
    #     if fw[i][j-1] > fw[i-1][j]:
    #         grid[i-1][j-2] = 0
    #         j -= 1

    #     else:
    #         grid[i-2][j-1] = 0
    #         i -= 1

    # for i in range(1, n+1)[::-1]:
    #     for j in range(1, n+1)[::-1]:
    #         if grid[i-1][j-1] == -1:
    #             bw[i][j] = float('-inf')
    #             continue
    #         bw[i][j] = max(bw[i+1][j], bw[i][j+1]) + (1 if grid[i-1][j-1] == 1 else 0)
    #         grid[i-1][j-1] = 0

    # res += fw[n][n] if fw[n][n] != float('-inf') else 0
    # res += bw[n][n] if bw[n][n] != float('-inf') else 0
    # return res


# p132
# dp: min cut at i 
# pal[j][i]: pal or not
def minCut(self, s):
    n = len(s)
    pal =[[False for _ in range(n)] for _ in range(n)]
    dp = [0]*(n)
    
    for i in range(n):
        tmp = i
        for j in range(i+1):
            if s[i]==s[j] and (j+1 > i-1 or pal[j+1][i-1]):
                pal[j][i] = True
                tmp = min(tmp, (dp[j-1]+1 if j-1>= 0 else 0))
        dp[i] = tmp
    return dp[-1]



# p516
# dp[i][j]: LPS for substring[i:j] inclusive
# dp[i][j] = dp[i-1][j-1]+2 if [i] == [j]
# else dp[i][j] = max(dp[i][j-1], dp[i+1][j])
# from the recursive relation we can see that the dp[i][j] is depending on next row and previous column, so we
# need to fill the row from bottom to the top
# all i > j, is set to 0 
def longestPalindromeSubseq(s):
    n = len(s)
    dp = [[0 for _ in range(n)] for _ in range(n)]
    for i in range(n)[::-1]:
        dp[i][i] = 1
        for j in range(i+1, n):
            if s[i] == s[j]:
                dp[i][j] = dp[i+1][j-1] + 2
            else:
                # either include s[i] or s[j]
                dp[i][j] = max(dp[i+1][j], dp[i][j-1])
    return dp[0][n-1]



""" implement recursive method with memoization """
def longestPalindromeSubseq(self, s):
    return self.longestPalSeq(0, len(s)-1, s, {})


def longestPalSeq(self, i, j, s, dic):
    if i > j: return 0 
    if i == j: return 1
    if (i, j) in dic: return dic[(i, j)]
    
    if s[i] == s[j]:
        ans = self.longestPalSeq(i+1, j-1, s, dic) + 2 
        dic[(i, j)] = ans
        return ans
    else:
        ans = max(self.longestPalSeq(i+1, j, s, dic), self.longestPalSeq(i, j-1, s, dic))
        dic[(i, j)] = ans
        return ans
            


# 647
# trick: expand each char and count the palindrones. meaning count all the palindrone containing [i] char and do this to all the chars.
def countSubstrings(self, s):
    total = 0
    for i in range(len(s)):
        total += self.expand(i, i, s) + self.expand(i, i+1, s)
    return total


def expand(self, left, right, s): 
    total = 0
    while left >= 0 and right < len(s) and s[left] == s[right]:
        left -= 1
        right += 1
        total += 1
    return total


# dp
# i represent all the substring start with i end at n-1 
# j, i <= j < n, check if s[i : j] is palindrome checking s[i] == s[j] and dp[i+1, j-1]
def countSubstrings(self, s):
    dp = [[False] * len(s) for _ in range(len(s))]
    cnt = 0
    for i in range(len(s)-1, -1, -1):
        for j in range(i, len(s)):
            dp[i][j] = s[i] == s[j] and (j-i+1 < 3 or dp[i+1][j-1]) # check if it is pal
            if dp[i][j]: cnt += 1
    return cnt
                        

# 5
# two pointer solution
# tick: each i as the center of palindrome, expand left and right and update the longest pal.
def longestPalindrome(self, s):
    ans =''
    for i in range(len(s)):
        tmp = self.expand(s, i, i)
        ans = max(tmp, ans, key=len)
        tmp = self.expand(s, i, i+1)
        ans = max(tmp, ans, key=len)
    return ans
    
        
def expand(self, s, l, r):
    while l >= 0 and r < len(s) and s[r] == s[l]:
        l -= 1
        r += 1
    return s[l+1 : r]


# TLE top-down recursive
def longestPalindrome_top_down(self, s):
    dic = {}
    return self.longest_pal(s, 0, len(s)-1, dic)


# return longest of pal of s 
def longest_pal(self, s, i, j, dic):
    if i == j: return s[i]
    if i > j: return ''
    
    if (i, j) in dic: 
        return dic[(i, j)]

    if s[i] == s[j] and  (s[i+1:j] in dic or self.pal(s, i+1, j-1, dic)):
        dic[(i, j)] = s[i:j+1]
        return s[i:j+1]
    else:
        a = self.longest_pal(s, i+1, j, dic)
        b = self.longest_pal(s, i, j-1, dic)
        
        if len(a) > len(b):
            dic[(i, j)] = a
            return a
        else:
            dic[(i, j)] = b
            return b 


def pal(self, s, i, j, dic):
    if s[i:j+1] == s[i:j+1][::-1]:
        dic[(i, j)] = s[i: j+1]
        return True
    return False



# TLE
def longestPalindrome(self, s):
    if len(s) <= 1: return s
    dp = [['' for _ in range(len(s))] for _ in range(len(s))]
    max_st = s[0]
    for i in range(len(s)-1, -1, -1):
        dp[i][i] = s[i]
        for j in range(i+1, len(s)):
            if dp[i+1][j-1] != None and s[i] == s[j]:
                dp[i][j] = s[i] + dp[i+1][j-1] + s[j]
                if len(max_st) < len(dp[i][j]): 
                    max_st = dp[i][j]
            else:
                dp[i][j] = None
    return max_st



""" AC DP solution """



# p410
# bottom-up
# dp[i][j]: minimum largest subarray sum for prefix i elements and j distinct subarrays
# dp[i][1] = sum of the prefix i elements
# dp[i][j] = min(dp[i][j], max(dp[k][j-1], sum([k]...[n-1])) , j <= i
def splitArray(nums, m):
    n = len(nums)
    dp = [[float('inf') for _ in range(m+1)]for _ in range(n+1)]
    total = [0]*(n+1)
    # initialize base case
    for i in range(1, n+1):
        total[i] = total[i-1] + nums[i-1]
        dp[i][1] = total[i]

    for i in range(1, n+1):
        for j in range(2, m+1):
            # only take care the case when subarrays acount is less or equal to array length
            if j <= i:
                for k in range(1, i):
                    # min([i][j], max(min_sum(0:k-1), sum(k: i)))
                    dp[i][j] = min(dp[i][j], max(dp[k][j-1], total[i] - total[k]))
    return dp[n][m]



# p368
# top down memo
def largestDivisibleSubset(self, nums: List[int]) -> List[int]:
    if not nums: return []
    nums.sort()
    memo = {}
    return max([self.findLDS(nums, i, memo) for i in range(len(nums))], key=len)


# find LDS ending with cur num
def findLDS(self, nums, cur, memo):
    if cur in memo: return memo[cur]
    tmpset = []
    for k in range(0, cur):
        if nums[cur] % nums[k] == 0: 
            tmp = self.findLDS(nums, k, memo)
            if len(tmp) > len(tmpset):
                tmpset = tmp
    memo[cur] = tmpset[::] + [nums[cur]]
    return memo[cur] 


# dp[i] longest divisible subset ending with [i]
# this is use more space than optimal solution, but idea is the same.
def largestDivisibleSubset(nums):
    if not nums: return []
    nums.sort()
    n = len(nums)
    dp = [[]]*(n+1)
    for i in range(1, n+1):
        dp[i] = [nums[i-1]]
        for k in range(1, i):
            # len(dp[i]) < len(dp[k])+1 to check if dp[i] already cover dp[k], if the length of dp[k] >= length dp[i], dp[i] needs to be updated
            if nums[i-1] % nums[k-1] == 0 and len(dp[i]) < len(dp[k])+1:
                # create a new list of element adding [i-1] to dp[k] and update to dp[i]
                dp[i] = dp[k] + [nums[i-1]]
    large = []
    for l in dp:
        if len(l) > len(large):
            large = l
    return large



# p256
# similar to path sum
def minCost(self, costs):
    if not costs:
        return 0
    for i in range(len(costs)-1)[::-1]:
        costs[i][0] += min(costs[i+1][1], costs[i+1][2])
        costs[i][1] += min(costs[i+1][0], costs[i+1][2])
        costs[i][2] += min(costs[i+1][0], costs[i+1][1])
    return min(costs[0])


# p276
# the problem is affacted by last two posts, if last two is same color or different colors.
# dp[i] is number ways to paint i posts using k colors where i <= n and no more than two ajacent posts can be same colors
# dp[i] = dp[i-2]*(k-1) + dp[i-1]*(k-1)
# just consider last three posts, we have two cases, one to make last two same and one to make last two different:
# case1: if previous two posts are the same, you have only one choice to make current post different than prevous two
# thus dp[i-1]*(k-1)
# case2: if previous two are different, we can make last two the same by dp[i-2]*(k-1)
# it can be optimized by reducing to using just three variable like fibnacci sum.


def numWays(n, k):
    if not n or not k:
        return 0
    if n < 2:
        return k
    dp = [1]*(n+1)
    dp[0], dp[1], dp[2] = 0, k, k*k
    for i in range(3, n+1):
        # 隔两个换一种颜色 和 隔一个换一种颜色的总和就是结果
        # 如果[i], [i-1] 是一个颜色，or [i]是一个颜色， i-2 前的posts有多少种涂法乘最后两种或一种的图法 就是 prefix i 的涂法。 
        dp[i] = dp[i-2]*(k-1) + dp[i-1]*(k-1)
    return dp[n]


# top down. 
def numWays(self, n: int, k: int) -> int:
        if n == 0 or k == 0: return 0 
        return self.numOfWays(n, k, -1, 0, {})
    
    
def numOfWays(self, n, colors, prev_color, last_two_same, dic):
    if n == 0: return 1
    if (n, last_two_same) in dic: return  dic[(n, last_two_same)]
    total = 0
    for color in range(1, colors+1):    
        if last_two_same and color == prev_color: continue
        total += self.numOfWays(n-1, colors, color, color == prev_color, dic)
    dic[(n, last_two_same)] = total
    return total
    


# p651
# dp[i]: max number of A's on the screen for i operations
# there are only two types of ending operations: ending with pressing A or ending with ctr + v (paste), and you allow to paste multiple times
# for n operations, min numbers of A's is n. so this is not the optimal solution, we can initialize the dp array with dp[i] = i
# the entire copy and paste takes minimum 3 operations, so additional 1 paste operation will append on copy to the end of A's on the screen
# i-k-1 meaning total length of A's after copy and paste.
# 多次paste比 重新来一次 select + copy +past 划算 
def maxA(N):
    if N <= 3:
        return N
    dp = [0]*(N+1)
    for i in range(N+1):
        dp[i] = i
        for k in range(1, i-2):
            dp[i] = max(dp[i], dp[k]*(i-k-1))
    return dp[N]


# p 264
# intutition: every ugly number is the product of smaller ugly number with one of [2,3,5] and the next ugly number must be the minimum of one of previous ugly number times [2,3,5].
# to decide which smaller previous ugly number to multiply with [2,3,5] to get next ugly number and at the same time avoid creating ugly numbers that
# are already in the list, thus we use three pointers p2, p3, p5 to track what is the next ugly number being used  to mutiply with [2,3,5]
# we increment each pointer whenever the ugly number if used and move onto the next ugly number.
# this procedure is similar with merge two sorted arrays.
def nthUglyNumber(n):
    ugly = [0]*n
    ugly[0] = 1
    p2, p3, p5 = 0, 0, 0
    for i in range(1, n):
        ugly[i] = min(ugly[p2]*2, ugly[p3]*3, ugly[p5]*5)
        if ugly[i] == ugly[p2]*2:
            p2 += 1
        if ugly[i] == ugly[p3]*3:
            p3 += 1
        if ugly[i] == ugly[p5]*5:
            p5 += 1
    return ugly[n-1]


# 343
# dp[i]: max product of multiple integers for prefix ith elements
# we notice that no matter how many integers you split into, we can all partition them into two parts
# so for j < i, dp[i] = max(max(j, dp[j]) * max(i-j, dp[i-j]))
# for some cases the j itself is greater than the dp[j] for example when j <= 4, in this case, we need to take the j as the number
# so we either take 1 number or multiple number to get the max product for the subproblem
def integerBreak(n):
    dp = [1]*(n+1)
    for i in range(2, n+1):
        for j in range(1, i):
            # the reason compare tot dp[i] here is because, from 1 to i partition can possiblly cause a increasing or decreasing of product. so we 
            # only need the max thus we alway compare to dp[i]
            dp[i] = max(dp[i], max(dp[j], j) * max(dp[i-j], i-j))
    return dp[n]



# 518
def change_memo(self, amount, coins):
    return self.num_comb(0, amount, coins, {})

def num_comb(self,start, amount, coins, dic): 
    if amount < 0: return 0
    if amount == 0: return 1
    if (start, amount) in dic: return dic[(start, amount)]
    ways = 0
    for i in range(start, len(coins)):
        ways += self.num_comb(i, amount - coins[i], coins, dic)
    dic[(start, amount)] = ways
    return ways
        


# tip: for prefix i coins, how many ways to get j-coins[i-1] amount will decide how my coins[i-1] will be repeated used. 
# thus it will also include the prefix i-1 coins, number of ways to get to j-coins[i-1] amount. 
# in other word, dp[i][j-coins[i-1]] will include dp[i-1][j-coins[i-1]]
def change_2d(self, amount, coins):
    dp = [[0 for _ in range(amount+1)] for _ in range(len(coins)+1)]
    dp[0][0] = 1
    for i in range(1, len(coins)+1):
        dp[i][0] = 1
        for j in range(1,amount+1):
            dp[i][j] = dp[i-1][j] + (dp[i][j-coins[i-1]] if j-coins[i-1] >= 0 else 0)
    return dp[len(coins)][amount]



# space reduce to one row
def change(self, amount, coins):
    dp = [0 for _ in range(amount+1)]
    dp[0] = 1
    for i in range(1, len(coins)+1):
        for j in range(1,amount+1):
            if j - coins[i-1] >= 0: 
                dp[j] = dp[j] + dp[j-coins[i-1]]
    return dp[amount]



# p983
# dp[i] is the min cost to cover ith day in 365 days
# if i day is not traval day, then dp[i] = dp[i-1]
# else dp[i] = min(dp[i-1]+costs[0], dp[i-7]+cost[1], dp[i-30]+costs[2])
# if i - day < 0, then dp[0] is used
def mincostTickets(self, days, costs):
    dp = [0] * 366
    dset = set(days)
    for i in range(1, 366):
        if i in dset:
            dp[i] = min(dp[i-1] + costs[0], dp[max(0, i-7)]+costs[1], dp[max(0, i-30)]+costs[2])
        else:
            dp[i] = dp[i-1]
            
    return dp[-1]


# p322
# top-down memo
# tip usually top-down memo need to have return value so that you can memo it.
def coinChange(self, coins, amount):
    self.dic = {}
    def dfs(amount):
        if amount < 0: return -1
        if amount == 0: return 0 
        if amount in self.dic: return self.dic[amount]
        
        cnt = float('inf')
        # amount and coin will be in dp definition
        for c in coins:
            t = dfs(amount-c)
            if t != -1:
                cnt = min(cnt, t+1)
        self.dic[amount] = cnt
        return cnt
    ans = dfs(amount)
    return ans if ans != float('inf') else -1  


# bottom up  
# similar to knapsack problem
def coinChange_2d(coins, amount):
    n = len(coins)
    dp = [[0 for _ in range(amount+1)] for _ in range(n+1)]
    for j in range(amount+1):
        dp[0][j] = float('inf')
    for i in range(1, n+1):
        for j in range(1, amount+1):
            if j - coins[i-1] >= 0:
                dp[i][j] = min(dp[i-1][j], dp[i][j-coins[i-1]]+1)
            else:
                dp[i][j] = dp[i-1][j]
    return dp[n][amount] if dp[n][amount] != float('inf') else -1



# reduced space solution
# dp[i]: fewest way to get i amount for given coins types where i <= amount
# dp[i] = min(dp[i], dp[i - a_coin] + 1)
def coinChange_1d(coins, amount):
    dp = [float('inf')] * (amount+1)
    # min ways to get 0 amount is 0
    dp[0] = 0
    for i in range(1, amount+1):
        for c in coins:
            if c <= i:
                dp[i] = min(dp[i], dp[i-c]+1)
    return dp[amount] if dp[amount] != float('inf') else -1



# 377
# DP top down memo
# states: target
# base: target == 0
# recur: num of comb = sum(num of comb(t-[i]))
def combinationSum4(self, nums: List[int], target: int) -> int:
    return self.comb(nums, target, {})
    
def comb(self, nums, target, dic):
    if target < 0: 
        return 0
    if target == 0: 
        return 1
    if target in dic:
        return dic[target]
    
    t = 0
    for n in nums:
        t += self.comb(nums, target - n, dic)
    dic[target] = t
    return t


# DP 
# permut(i): number of permutations for target value i
# permut[i] = sum(permut[i- [j]) where i >= [j]
def combinationSum4(nums, target):
    permut = [0]*(target+1)
    permut[0] = 1
    for i in range(1, target+1):
        for n in nums:
            if i >= n:
                permut[i] += permut[i-n]
    return permut[target]



# p300
# dp[i]: longest increasing sequence ending with ith element
# we know that the longest incresing squence must end with one of the element in the array, so we can come up with the above
# definition. we only need to think a little diffrent than standard dp problem for example we will think dp[i-1] for standard dp problem but
# this problem demonstrates that subproblem not necessarily be the previous one subproblem, it can be in any one of the
# previous subproblemc.
def lengthOfLIS(nums):
    n = len(nums)
    dp = [1]*n
    for i in range(1, n):
        for j in range(i):
            if nums[i] > nums[j]:
                tmp = dp[j] + 1
                dp[i] = max(dp[i], tmp)
    return max(dp)



# p673
# same as above problem and add a simple problem to it
# LIS[i]: length of longest increasing subsequneces ending with [i]
# cnt[i]: number of longest increasing subsequences ending with [i]
def findNumberOfLIS(nums):
    if not nums:
        return 0
    res = 0
    n = len(nums)
    leng = [1]*n
    cnt = [1]*n
    for i in range(1, n):
        for j in range(i):
            if nums[i] > nums[j]:
                # find all the subsequence that is same length of current max length ending i
                if leng[i] == leng[j] + 1:
                    cnt[i] += cnt[j]
                elif leng[i] < leng[j] + 1:
                    leng[i] = leng[j]+1
                    cnt[i] = cnt[j]

    max_leng = max(leng)
    for i in range(n):
        if leng[i] == max_leng:
            res += cnt[i]
    return res


# p416
#  top-down memo
def canPartition(self, nums):
    total = sum(nums)
    if total & 1 != 0: return False
    target = total // 2
    if max(nums) > target: return False
    return self.dfs(0, nums, target, {})


def dfs(self, idx, nums, target, seen):
    if target < 0: return False
    if target == 0: return True 
    if (idx, target) in seen: return seen[(idx, target)]
    for i in range(idx, len(nums)):
        if self.dfs(i+1, nums, target-nums[i], seen): 
            return True
    seen[(idx, target)] = False
    return False 



# dp - knapsack problem
def canPartition(nums):
    n = len(nums)
    if n < 2: return False
    total = sum(nums)
    if total % 2 != 0: return False
    target = total//2

    dp = [[False for _ in range((total//2)+1)] for _ in range(n+1)]
    dp[0][0] = True

    for i in range(1, n+1):
        for j in range(1, target+1):
            dp[i][j] = dp[i-1][j] or (dp[i-1][j-nums[i-1]] if j - nums[i-1] >= 0 else False)

    return dp[n][total//2]


# space compression solution 
# because the result is only related to [i-1][j] and [i-1][j - nums[i-1]]
# we can use a prefix i loop to represent current row. and dp[w] to represent whether prefix i elements can fill the w capacity. 
# dp[j] = dp[j] || dp[j-nums[i-1]] where j: w->nums[i-1] inclusive decrement. 
# because we are using only one row, but we are imaging there are 2-d arrays, we merge all row's result(except the first row) into the first row. 
# the reason we reverse the looping order is because, imaging 2-d array, current cell is depending on the cell that is located at previous colon one row above current cell. 
# if we are looping forward, the above cell will be updated, but it should not be updated until the latter cell that is depending on its original value. 
# thuse we do it backward, so the the latter cell get the original value and then it updates the value. 

def canPartition_space_optimized(self, nums):
    n = len(nums)
    if n < 2: return False
    total = sum(nums)
    if total % 2 != 0: return False
    target = total//2
    
    dp = [False] * (target+1)
    dp[0] = True 
    
    for i in range(1, n+1):
        for j in range(target, nums[i-1]-1, -1):
            dp[j] = dp[j] or dp[j-nums[i-1]]
    
    return dp[-1]




# 10
# recursion:
# we know that * matches empty string or more previous char
# consider two cases:
# if first char in string and pattern is matched, and second char in pattern is not '*',
# then recursively compare the rest substrings in string and pattern
# if first char is matched in string and pattern and the sudo xcodebuild -licensesecond char in pattern is '*', then recursively compare rest of string with the pattern.
# consider edge case, when the pattern starts with a space and followed by a '*', we know stars means that there is 0 or more
# prev char. so if could be 0 space. thus we recursively compare the string and rest of substring after the star.


def isMatch(s, p):
    if not p:
        return not s
    first = (len(s) != 0 and s[0] == p[0] or p[0] == '.')
    if len(p) >= 2 and p[1] == '*':
        return isMatch(s, p[2:]) or (first and isMatch(s[1:], p))
    else:
        return first and isMatch(s[1:], p[1:])


# dynamic programming solution
# dp[i][j] if s[:i] and p[:j] is matched or not
# we separate problem into two cases: p[j-1] is * or not
# if p[i-1] is not *, then compare the p[j-1] and s[i-1] and dp[i-1][j-1]
# else we take the result of three cases when * takes 0, 1, or >1 repetitions of previous char
# edge case: when s is empty, we can use #*#*#* ... to match empty string, so we can initialize the result in the dp table
def isMatch_star(s, p):
    if s == None or p == None:
        return False
    n = len(s)
    m = len(p)
    dp = [[False for _ in range(m+1)] for _ in range(n+1)]
    dp[0][0] = True
    # for the case when s is empty, and possible patterns to match empty string
    for j in range(2, m+1):
        if p[j-1] == '*' and dp[0][j-2]:
            dp[0][j] = True

    for i in range(1, n+1):
        for j in range(1, m+1):
            if s[i-1] == p[j-1] or p[j-1] == '.':
                dp[i][j] = dp[i-1][j-1]
            else:
                if p[j-1] == '*':
                    if s[i-1] != p[j-2] and p[j-2] != '.':
                        dp[i][j] = dp[i][j-2]
                    else:
                        # 0, 1, or more
                        dp[i][j] = dp[i][j-2] or dp[i][j-1] or dp[i-1][j]
                # rest of cases would be false
    return dp[n][m]



# p44
def isMatch_wildcard(s, p):
    if not s and not p:
        return True
    # code block below is just simplify the pattern string becuase a**a is the same as a*a
    np, i, = [], 0
    while i < len(p):
        if p[i] != '*':
            np.append(p[i])
            i += 1
        else:
            np.append('*')
            while i < len(p) and p[i] == '*':
                i += 1

    n = len(s)
    m = len(np)
    dp = [[False for _ in range(m+1)] for _ in range(n+1)]
    dp[0][0] = True

    # edge case when first char is star
    if m > 0 and np[0] == '*':
        dp[0][1] = True

    for i in range(1, n+1):
        for j in range(1, m+1):
            if s[i-1] == np[j-1] or np[j-1] == '?':
                dp[i][j] = dp[i-1][j-1]
            else:
                # recursive function, star takes empty or more chars
                if np[j-1] == '*':
                    dp[i][j] = dp[i-1][j] or dp[i][j-1]
    return dp[n][m]


""" game theory / DP / minimax """
# minimax assume both players play optimally but sometimes it can be misunderstood that play optimally meaning play greedy.
# this is not right, play will choose the right one but the currrent bigger one.
# p 486
# brute force using recursion
# there is two cases, player1 choose left or right of current array
# and the opponenet will try to get as much as he can in the rest array.
# TLE O(2^N)


def PredictTheWinner_BF(nums):
    def get_score_diff(nums, l, r):
        if l == r:
            return nums[l]
        return max(nums[l] - get_score_diff(nums, l+1, r), nums[r] - get_score_diff(nums, l, r-1))
    return get_score_diff(nums, 0, len(nums)-1) >= 0


# the problem states that two player will both try to get optimal scores for the given array.
# because the toal score is fixed, the problem is asking whehter player1 can win. so we need to try to maximize player1 score
# dp[i][j]: is how much player1 is win over player2 in the subarray [i:j]
# the main idea is the same as above recursive, just start from bottom up.
# this is another problem that the result is at the top right corner, last one is the palindrome problem also with two index i , j to indicate first and last.
# this type of questions usually need to fill the table from bottom to the top.
def PredictTheWinner(nums):
    n = len(nums)
    dp = [[0 for _ in range(n+1)] for _ in range(n+1)]
    for i in range(n):
        dp[i][i] = nums[i]
    for i in range(n)[::-1]:
        for j in range(i, n):
            # if player1 choose i element and choose j element, the how much opponent can win over player1 for the rest array.
            dp[i][j] = max(nums[i] - dp[i+1][j], nums[j] - dp[i][j-1])
    return dp[0][n-1] >= 0


# p 464
# top down dp
def canIWin(maxChoosableInteger, desiredTotal):
    if (maxChoosableInteger+1)*maxChoosableInteger//2 < desiredTotal:
        return False
    memo = {}

    def win_helper(nums, target):
        # use nums' string as the hash key for memo dic
        hs = str(nums)
        if hs in memo:
            return memo[hs]
        # base case: if the last one number is at least target, wins.
        if nums[-1] >= target:
            return True
        for i in range(len(nums)):
            # if opponent cannot win if player 1 pick i element, than player1 is winning
            # expecting opponent to be false so I can win.
            if not win_helper(nums[:i]+nums[i+1:], target-nums[i]):
                memo[hs] = True
                return True
        # if for current array, player1 cannot win, then store in memo, so next time encounter the same sequence will return directly.
        memo[hs] = False
        return False
    return win_helper(range(1, maxChoosableInteger+1), desiredTotal)



# 1049
# dp[i][j]: whether for prefix i, can sum to j
# s1 + s2 = S and s1- s2 = diff. diff = S-2*s2
# if you want to find the min of diff, you need to find the max of s2
# and the s2 is in the range of [0, s/2], thus j is in this range, and i in [0, n]
# dp[i][0] = True
def lastStoneWeightII(stones):
    total = sum(stones)
    n = len(stones)
    dp = [[False for _ in range(total+1)] for _ in range(n+1)]
    for i in range(n+1):
        dp[i][0] = True
    local = float('-inf')
    for i in range(1, n+1):
        for j in range(1, total//2+1):
            # keep in mind that when never there is some add or subtract when accessing array element, make sure that the result is not negative, check before accessing.
            if (j-stones[i-1] >= 0 and dp[i-1][j-stones[i-1]]) or dp[i-1][j]:
                dp[i][j] = True
                local = max(local, j)
    return total - 2 * local



# p1048
# dp[i]: the longest string chain ending with ith word
def longestStrChain(words):
    word_group = {i: set() for i in range(1, 17)}
    for w in words:
        word_group[len(w)].add(w)
    # this is way to set default value for each key in a dictionary.
    dp = collections.defaultdict(lambda: 1)
    for i in range(2, 17):
        for w in word_group[i]:
            for j in range(i):
                prev = w[:j] + w[j+1:]
                if prev in word_group[i-1]:
                    dp[w] = max(dp[w], dp[prev] + 1)
    return max(dp.values() or [1])


def longestStrChain_short(words):
    dp = {}
    # sorting words array using their length
    for w in sorted(words, key=len):
        tmp = 0
        for j in range(len(w)):
            tmp = max(dp.get(w[:j] + w[j+1:], 0) + 1, tmp)
        dp[w] = tmp
    return max(dp.values())




# 494
# dfs - memo  TLE 
# key: the store the key in memo needs to represent unique state.
def findTargetSumWays(self, nums, S):
    return self.dfs(0, nums, S, {})
        
def d_fs(self, idx, nums, target, dic):
    if idx == len(nums):
        if target == 0: 
            return 1
        return 0 
    
    if (idx, target) in dic and dic[(idx, target)] > 0: return dic[(idx, target)]
    
    neg = self.dfs(idx+1, nums, target + nums[idx], dic)
    pos = self.dfs(idx+1, nums, target - nums[idx], dic)        
    dic[(idx, target)] = neg + pos
    return neg + pos




""" OOP design problems """

# p146
# least recent used cache
# we are using a dictionary and double linked list to implement LRU for O(1) operation.
# intuition: whenever we perform get or put operation, that key will become most recent used key, so we will need to put
# it in the end of list, and the staled keys are in front of list. So for put operation, we need to put current value node in the end of the list
# if the capacity is exceeded, we just remove the front of the list. these operation will take O(1), because of the feature of
# double linkedlist, it can add or remove itself using O(1) time.
# for the get operation, if the key is not in the dictionary, return -1, else we will remove the k corresponding node from the list
# and add this node to the end of the list, this node become most recently used node.
#
# we need a tail and head node to indicate the boundry of the list

# double linkedlist node


class Node:
    def __init__(self, k, v):
        # every node will store the k of itself in the dictionary
        self.key = k
        self.val = v
        self.prev = None
        self.next = None


class LRUCache:

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.dic = {}
        self.head = Node(0, 0)
        self.tail = Node(0, 0)
        self.head.next = self.tail
        self.tail.prev = self.head

    def get(self, key: int) -> int:
        if key in self.dic:
            node = self.dic[key]
            # remove the node from its original place
            self._remove(node)
            # add the node to the end
            self._add(node)
            return node.val
        return -1

    def put(self, key: int, value: int) -> None:
        if key in self.dic:
            self._remove(self.dic[key])
        node = Node(key, value)
        self.dic[key] = node
        self._add(node)
        if len(self.dic) > self.capacity:
            n = self.head.next
            # if exceed the cap, remove the first node (head.next)
            self._remove(n)
            # remove the entry in dic
            del(self.dic[n.key])

    def _remove(self, node: object) -> None:
        prev = node.prev
        nxt = node.next
        prev.next = nxt
        nxt.prev = prev

    def _add(self, node: object) -> None:
        prev = self.tail.prev
        node.prev = prev
        prev.next = node
        node.next = self.tail
        self.tail.prev = node


# 341
# tip: this is assumed for implementing iterator, usually hasNext() will be called before next() is called. 
class NestedIterator(object):

    def __init__(self, nestedList):
        """
        Initialize your data structure here.
        :type nestedList: List[NestedInteger]
        """
        self.stack = nestedList[::-1]

    def next(self):
        """
        :rtype: int
        """
        return self.stack.pop().getInteger()
        

    def hasNext(self):
        """
        :rtype: bool
        """
        while self.stack:
            top = self.stack[-1]
            if top.isInteger():
                return True
            # flaten the list until find the first avaible integer.
            self.stack = self.stack[:-1] + top.getList()[::-1]
        return False
       


# 284
# trick use a variable to store next element.
class PeekingIterator(object):
    def __init__(self, iterator):
        """
        Initialize your data structure here.
        :type iterator: Iterator
        """
        self.iter = iterator
        self.nxt = self.iter.next() if self.iter.hasNext() else None
        

    def peek(self):
        """
        Returns the next element in the iteration without advancing the iterator.
        :rtype: int
        """
        return self.nxt
        

    def next(self):
        """
        :rtype: int
        """
        ans = self.nxt
        self.nxt = self.iter.next() if self.iter.hasNext() else None
        return ans
        

    def hasNext(self):
        """
        :rtype: bool
        """
        return self.nxt != None


# 635
# trick is using map to map out the granularity and compare each pair of timestamp
class LogSystem(object):

    def __init__(self):
        self.g = {'Year': 5, 'Month': 8, 'Day':11, 'Hour':14, 'Minute':17, 'Second':20}
        self.times = []
        

    def put(self, id, timestamp):
        """
        :type id: int
        :type timestamp: str
        :rtype: None
        """
        self.times.append([id, timestamp])
        

    def retrieve(self, s, e, gra):
        """
        :type s: str
        :type e: str
        :type gra: str
        :rtype: List[int]
        """
        idx = self.g[gra]
        start = s[:idx]
        end = e[:idx]
        
        return [i for i, t in self.times if start <= t[:idx] <= end]




# p155
# idea: put current min on the top and update the min along the push operations
class MinStack:

    def __init__(self):
        self.stack = []

    def push(self, x: int) -> None:
        if not self.stack or x < self.stack[-1]:
            self.stack.append(x)
            self.stack.append(x)
        else:
            cur_min = self.stack[-1]
            self.stack.append(x)
            self.stack.append(cur_min)

    def pop(self) -> None:
        self.stack.pop()
        self.stack.pop()

    def top(self) -> int:
        return self.stack[-2]

    def getMin(self) -> int:
        return self.stack[-1]



# 716
class MaxStack(object):

    def __init__(self):
        """
        initialize your data structure here.
        """
        self.stack = []
        

    def push(self, x):
        """
        :type x: int
        :rtype: None
        """
        if self.stack:
            cur_max = max(x, self.stack[-1][1]) 
            self.stack.append((x, cur_max))
        else:
            self.stack.append((x, x))
        

    def pop(self):
        """
        :rtype: int
        """
        return self.stack.pop()[0]

    def top(self):
        """
        :rtype: int
        """
        return self.stack[-1][0]
        

    def peekMax(self):
        """
        :rtype: int
        """
        return self.stack[-1][1]

    def popMax(self):
        """
        :rtype: int
        """
        cur_max = self.stack[-1][1]
        tmp = []
        # find the entry that has the max_value
        while self.stack[-1][0] != cur_max:
            tmp.append(self.stack.pop())
        
        self.stack.pop()
        while tmp: 
            # trick here is use the push function to put old entry back to the stack when you remove the current max.
            self.push(tmp.pop()[0])
        
        return cur_max


# 170
class TwoSum(object):

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.nums = {}
        

    def add(self, number):
        """
        Add the number to an internal data structure..
        :type number: int
        :rtype: None
        """
        if number in self.nums: 
            self.nums[number] += 1
        else:
            self.nums[number] = 1

    def find(self, value):
        """
        Find if there exists any pair of numbers which sum is equal to the value.
        :type value: int
        :rtype: bool
        """
        nums = self.nums
        
        for n in nums:
            # if in the map you can find the complement and the complement is not the same as the current n or current n has many dup. 
            if value - n in nums and (value - n != n or nums[value-n] > 1):
                return True 
        return False
        
        
# 604
class StringIterator(object):

    def __init__(self, compressedString):
        """
        :type compressedString: str
        """
        self.st = compressedString
        self.queue = collections.deque()
        i, n = 0, len(self.st)
        
        while i < n:
            j = i + 1
            # find number of time the i char is repeated
            while j < n and self.st[j].isdigit(): j += 1
            # store [char, repeat] in a queue
            self.queue.append([self.st[i], int(self.st[i+1: j])])
            i = j
     

    def next(self):
        """
        :rtype: str
        """
        if self.queue:
            front = self.queue[0]
            front[1] -= 1 
            # if current char has not more repeat, pop it off.
            if front[1] == 0: 
                self.queue.popleft()
            return front[0]
        return ' '
            
            
    def hasNext(self):
        """
        :rtype: bool
        """
        return len(self.queue) != 0 



#362  think each time slot as a bucket and all the hits happenedi the slot will put into this slot.
class HitCounter(object):

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.time_bucket = [(0, 0)] * 300 
        

    def hit(self, timestamp):
        """
        Record a hit.
        @param timestamp - The current timestamp (in seconds granularity).
        :type timestamp: int
        :rtype: None
        """
        idx = timestamp % 300 
        time, hits = self.time_bucket[idx] 
        
        if timestamp != time:
            self.time_bucket[idx] = timestamp, 1
        else:
            self.time_bucket[idx] = time, hits+1
        
        

    def getHits(self, timestamp):
        """
        Return the number of hits in the past 5 minutes.
        @param timestamp - The current timestamp (in seconds granularity).
        :type timestamp: int
        :rtype: int
        """
        ans = 0 
        
        for time, hits in self.time_bucket :
            if timestamp - time < 300:
                ans += hits
        return ans
         


# 218 
# trick: pop one off current list and add the current list to the end of the queue.
class ZigzagIterator(object):

    def __init__(self, v1, v2):
        """
        Initialize your data structure here.
        :type v1: List[int]
        :type v2: List[int]
        """
        self.queue = collections.deque()
        if v1: 
            self.queue.append(collections.deque(v1)) 
        if v2: 
            self.queue.append(collections.deque(v2)) 
        

    def next(self):
        """
        :rtype: int
        """
        cur = self.queue.popleft()
        res = cur.popleft()
        if cur:
            self.queue.append(cur)
        return res 


    def hasNext(self):
        """
        :rtype: bool
        """
        if self.queue: return True
        return False


# 348 
# space O(n^2), time O(n)
class TicTacToe(object):

    def __init__(self, n):
        """
        Initialize your data structure here.
        :type n: int
        """
        self.board = [[0 for _ in range(n)] for _ in range(n)]
        

    def move(self, row, col, player):
        n = len(self.board)
        if row >= n or col >= n: return 0 
        if self.board[row][col] != 0: return 0 
        self.board[row][col] = player if player == 1 else 2 
        
        if self.check_horizontal(row, player): return player
        if self.check_vertical(col, player): return player 
        if self.check_diagnal(row, col, player): return player
        
        return 0 
        
    
    def check_horizontal(self, r, player):
        for i in range(len(self.board)):
            if self.board[r][i] != player: return False
        return True
       
    
    def check_vertical(self, c, player):
        for i in range(len(self.board)):
            if self.board[i][c] != player: return False
        return True
       
        
    def check_diagnal(self, r, c, player):
        n = len(self.board)
        if r != c and (r + c) != n-1:
            return False
        
        top_left_bottom_right = True
        top_right_bottom_left = True 
        for i in range(n):
            if self.board[i][i] != player: 
                top_left_bottom_right = False
                break
                
        for j in range(n):
            if self.board[j][n-1-j] != player:
                top_right_bottom_left = False
                break
                
        return top_left_bottom_right or top_right_bottom_left


# optimal
# anti_diagnal pattern: for n*n grid, if i+j == n-1, (i, j) is on the anti diagnal.
class TicTacToe(object):

    def __init__(self, n):
        self.row = [0] * n
        self.col = [0] * n
        s# because there are only two diagnals that can form winning condition.
        self.diagnal = 0 
        self.anti_diagnal = 0 
        
        
    # Observation:if both player has placed a step on the same row, no one can win on this row
    # they are blocking each other. same logic apply to col, diagnal and anti diaganl.
    def move(self, r: int, c: int, player: int) -> int:
        n = len(self.row)
        self.row[r] += 1 if player == 1 else -1
        self.col[c] += 1 if player == 1 else -1 
        if r == c: self.diagnal += 1 if player == 1 else -1
        if r + c == n-1: self.anti_diagnal += 1 if player == 1 else -1
        if self.row[r] == n or self.col[c] == n or self.diagnal == n or self.anti_diagnal == n:
            return 1 
        if self.row[r] == -n or self.col[c] == -n or self.diagnal == -n or self.anti_diagnal == -n:
            return 2
        return 0
        



# p173
# just iterative in order traversal
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


class BSTIterator:
    # only add left nodes to the stack to avoid store all the nodes into the memory 
    def __init__(self, root: TreeNode):
        self.stack = []
        self._push_all(root)

    def _push_all(self, node: TreeNode):
        while node:
            self.stack.append(node)
            node = node.left

    def next(self) -> int:
        """
        @return the next smallest number
        """
        nxt = self.stack.pop()
        self._push_all(nxt.right)
        return nxt.val

    def hasNext(self) -> bool:
        """
        @return whether we have a next smallest number
        """
        return self.stack




# p232
# two stack == queue
# tricky part is when you need to push and peek again if you previously already pop all items off the input.
class MyQueue:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.input = []
        self.output = []

    def push(self, x: int) -> None:
        """
        Push element x to the back of queue.
        """
        self.input.append(x)

    def pop(self) -> int:
        """
        Removes the element from in front of queue and returns that element.
        """
        self.peek()
        return self.output.pop()

    def peek(self) -> int:
        """
        Get the front element.
        """
        # only pop all the input into output when the output is empty
        if not self.output:
            while self.input:
                self.output.append(self.input.pop())
        return self.output[-1]

    def empty(self) -> bool:
        """
        Returns whether the queue is empty.
        """
        return not self.input and not self.output




# 225
class MyStack(object):
    
    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.q = collections.deque()
        

    def push(self, x):
        """
        Push element x onto stack.
        :type x: int
        :rtype: None
        """
        # mvoe the existing element after the newly added element
        n = len(self.q)
        self.q.append(x)
        while n > 0:
            self.q.append(self.q.popleft()) 
            n -= 1
        

    def pop(self):
        """
        Removes the element on top of the stack and returns that element.
        :rtype: int
        """
        return self.q.popleft()
        

    def top(self):
        """
        Get the top element.
        :rtype: int
        """
        return self.q[0]
        

    def empty(self):
        """
        Returns whether the stack is empty.
        :rtype: bool
        """
        return not self.q




# p380
# intuition: return random element indicate that we need an array like data structure. O(1) checking if an element is in the set, using hashmap/dictionary
# solution: use a dictionary to store the value as the key and the index of the value in an array as the value.
# dict (val, index) <--> array(index, val)
# follow up: if dup is allowed, then solution is dic(val, set of index) <--> array(index, val)


# problem ask for a set, its not ordered so no need to use doublelinked list node to solve it.
class RandomizedSet:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        # to be able to return random set element, we can use a random access datastructure like list in python
        self.dic = {}
        self.arr = []

    def insert(self, val: int) -> bool:
        """
        Inserts a value to the set. Returns true if the set did not already contain the specified element.
        """
        if val in self.dic:
            return False
        # add the index of the element in the array as the value of the key
        self.dic[val] = len(self.arr)
        # add the end of the arr
        self.arr.append(val)
        return True

    # tricky part is to swap the element needed to be removed with the last element.
    # careful with the swaping 
    def remove(self, val: int) -> bool:
        """
        Removes a value from the set. Returns true if the set contained the specified element.
        """
        if val not in self.dic:
            return False
        idx = self.dic[val]
        # make sure removing element is not the last one.
        last_idx = len(self.arr) - 1
        last = self.arr[last_idx]
        if idx < last_idx:
            self.arr[idx], self.arr[last_idx] = self.arr[last_idx], self.arr[idx]
            # do not forget to update dic
            self.dic[val], self.dic[last] = last_idx, idx
        # remove the last item and remove the entry in the dic
        self.arr.pop()
        del(self.dic[val])
        return True

    def getRandom(self) -> int:
        """
        Get a random element from the set.
        """
        return self.arr[random.randint(0, len(self.arr)-1)]



# 251 
# tip: when you implememnt iterator, do not implement heavy logic inside constructor because purpose of iterator is to save memory when the object is too
# to be entirely loaded into meomry. so dont just copy all the element into an array and making a list iterator. 
# trick: when the inner reach the end of the list, reset to 0 until the outer reach the end
class Vector2D(object):

    def __init__(self, v):
        """
        :type v: List[List[int]]
        """
        self.v = v
        self.inner = 0 
        self.outer = 0
    

    def advance_to_next(self):
        while self.outer < len(self.v) and self.inner == len(self.v[self.outer]):
            self.inner = 0 
            self.outer += 1
        
        
    def next(self):
        """
        :rtype: int
        """
        self.advance_to_next()
        ans = self.v[self.outer][self.inner]
        self.inner += 1
        return ans 


    def hasNext(self):
        """
        :rtype: bool
        """
        # skip empty list 
        self.advance_to_next()
        return self.outer < len(self.v)




# p706
# using linkedlist to avoid collision.

class ListNode:
    def __init__(self, k, v):
        self.key_val_pair = (k, v)
        self.next = None


class MyHashMap:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        # number of hashtable slots
        self.slots = 10000
        self.ht = [None]*self.slots

    def put(self, key: int, value: int) -> None:
        """
        value will always be non-negative.
        """
        idx = key % self.slots
        if not self.ht[idx]:
            self.ht[idx] = ListNode(key, value)
        else:
            cur = self.ht[idx]
            while True:
                if cur.key_val_pair[0] == key:
                    cur.key_val_pair = (key, value)
                    return
                if not cur.next:
                    break
                cur = cur.next
            cur.next = ListNode(key, value)

    def get(self, key: int) -> int:
        """
        Returns the value to which the specified key is mapped, or -1 if this map contains no mapping for the key
        """
        idx = key % self.slots
        cur = self.ht[idx]
        while cur:
            if cur.key_val_pair[0] == key:
                return cur.key_val_pair[1]
            cur = cur.next
        return -1

    def remove(self, key: int) -> None:
        """
        Removes the mapping of the specified value key if this map contains a mapping for the key
        """
        idx = key % self.slots
        cur, prev = self.ht[idx], self.ht[idx]
        if not cur:
            return
        if cur.key_val_pair[0] == key:
            self.ht[idx] = cur.next
        else:
            cur = cur.next
            while cur:
                if cur.key_val_pair[0] == key:
                    prev.next = cur.next
                    break
                else:
                    cur, prev = cur.next, prev.next



# 244
# trick is the list is sorted in the dict you can use same technique like merge two sorted arrays.
class WordDistance(object):

    def __init__(self, words):
        """
        :type words: List[str]
        """
        self.dic = collections.defaultdict(list)
        for i in range(len(words)):
            self.dic[words[i]].append(i)
        

    def shortest(self, word1, word2):
        list1 = self.dic[word1]
        list2 = self.dic[word2]
        
        n, m = len(list1), len(list2)
        i, j = 0, 0
        min_dist = float('inf')
        while i < n and j < m:
            if list1[i] < list2[j]:
                min_dist = min(list2[j] - list1[i], min_dist)
                i += 1
            else:
                min_dist = min(list1[i] - list2[j], min_dist)
                j += 1
        
        return min_dist
        
        



# 364 
# descriptive way
class MovingAverage(object):

    def __init__(self, size):
        """
        Initialize your data structure here.
        :type size: int
        """
        self.size = size
        self.queue = collections.deque()
        self.total = 0


    def next(self, val):
        """
        :type val: int
        :rtype: float
        """
        queue = self.queue
        queue.append(val)
        self.total += val
        if len(queue) <= self.size:
            return float(self.total) / len(queue)
        else:
            self.total -= queue.popleft()
            return float(self.total) / self.size


# pythonic way 
class MovingAverage_(object):

    def __init__(self, size):
        """
        Initialize your data structure here.
        :type size: int
        """
        self.size = size
        self.queue = collections.deque(maxlen=size) # only allow fix size of items


    def next(self, val):
        """
        :type val: int
        :rtype: float
        """
        queue = self.queue
        queue.append(val)
        # can use sum(  )
        return float(sum(queue)) / len(queue)

        

# 359 
# naive solution, because you need to store all the messages in the memeory.
class Logger(object):

    def __init__(self):
        self.map = {}
        

    def shouldPrintMessage(self, timestamp, message):
        map = self.map
        if message not in map:
            map[message] = timestamp
            return True
        else:
            if timestamp - map[message] < 10:
                return False
            map[message] = timestamp
            return True
        


# optimal and practical solution using priority queue.
import heapq 
class Logger_(object):
    def __init__(self):
        self.msg_heap = []
        self.seen = set()
        

    def shouldPrintMessage(self, timestamp, message):
        msg_heap = self.msg_heap
        while msg_heap and timestamp - msg_heap[0][0] >= 10:
            _ , msg = heapq.heappop(msg_heap)
            self.seen.remove(msg)

        if message not in self.seen:
            self.seen.add(message)
            heapq.heappush(msg_heap, (timestamp, message))
            return True 
        return False
       


""" trie """
# 1268
# trie is a tree structue, for some node in the tree
# there is only one path that can reach this node due to the reason that in tree 
# structure, each tree node will have one and only one parent. 

class TrieNode:
    def __init__(self):
        # char -> TrieNode
        self.children = {}
        # add the word sources current path characters.
        self.words = []      
        
class Trie:
    def __init__(self):
        self.root = TrieNode()
    
    
    def insert(self, word):
        r = self.root
        for c in word:
            if c not in r.children:
                r.children[c] = TrieNode()
            # update r to c's trie node
            r = r.children[c]
            # add what word is c in into to the words list
            r.words.append(word)
            r.words.sort()
            # pop if list just over three
            # the result should only keep the top three lexi order words. 
            if len(r.words) > 3: 
                r.words.pop()
        
    
    # return search results for each prefix of the word 
    def search(self, word):
        r = self.root
        res = []
        for i, c in enumerate(word):
            if c not in r.children:
                res += [[] for _ in range(len(word)-i)]
                return res
            r = r.children[c]
            res += [r.words]
        return res
        
        

class Solution:
    def suggestedProducts(self, products: List[str], searchWord: str) -> List[List[str]]:
        trie = Trie()
        for p in products:
            trie.insert(p)
        return trie.search(searchWord)
        

# best solution + optimal
# binary search
# search space: sorted words list
# key observation: in sorted list, if words[i] is prefix if words[j], then,
# words[j] is prefixed with words[i+1], words[i+2] .. words[j].
# search goal: searh for insertion point of each prefix of given word, and expand 3 words ahead
# check if these 3 are starts with this prefix
# source code biserct_left.   https://docs.python.org/3.7/library/bisect.html
def suggestedProducts(self, products: List[str], searchWord: str) -> List[List[str]]:
    products.sort()
    res, prefix = [], ''
    for c in searchWord:
        prefix += c
        i = bisect.bisect_left(products, prefix, 0)
        res.append([w for w in products[i: i+3] if w.startswith(prefix)])
    return res 
            



# 981
class TimeMap:
    def __init__(self):
        self.dic = collections.defaultdict(list)
        

    def set(self, key: str, value: str, timestamp: int) -> None:
        self.dic[key].append((timestamp, value))
        

    def get(self, key: str, timestamp: int) -> str: 
        if key not in self.dic: return ''
        return self.search(self.dic[key], timestamp)
    
#   find the first time > given time 
    def search(self, values, target):
        left, right = 0, len(values)
        while left < right:
            mid = (left + right)//2
            
            if values[mid][0] <= target:
                left = mid + 1
            else:
                right = mid 
                
        return values[left-1][1] if 0 <= left - 1 < len(values) else ''

     



# 208
# trie is tree like data structure, it is used to store characters for in each node, with ending signal when a word is reached
# the trie node may include is_word flag to indicate current word is or isnot a word, and a map to its neighbours.
# trie class have insert, search and startwith methods

# simple trie implementation using dic. does not recomend in interview coz dic key map to differnt type of data
class Trie(object):

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.trie = {}
        

    def insert(self, word):
        """
        Inserts a word into the trie.
        :type word: str
        :rtype: None
        """
        trie = self.trie
        for w in word:
            if w not in trie:
                trie[w] = {}
            trie = trie[w] 
        trie["#"] = '#'
        

    def search(self, word):
        """
        Returns if the word is in the trie.
        :type word: str
        :rtype: bool
        """
        trie = self.trie
        for w in word:
            if w not in trie:
                return False
            trie = trie[w]
        if '#' in trie: 
            return True 
        return False
        

    def startsWith(self, prefix):
        """
        Returns if there is any word in the trie that starts with the given prefix.
        :type prefix: str
        :rtype: bool
        """
        trie = self.trie
        for w in prefix:
            if w not in trie:
                return False
            trie = trie[w]
        
        return True


# better implementation for interview purpose.
class Node:
    def __init__(self, c=''):
        self.char = c
        self.is_word = False
        self.subtries = {}    
    

class Trie:
    def __init__(self):
        self.trie = Node(' ')
        

    def insert(self, word: str) -> None:
        cur_trie = self.trie
        for c in word:
            if c not in cur_trie.subtries:
                cur_trie.subtries[c] = Node(c)
            cur_trie = cur_trie.subtries[c]
        cur_trie.is_word = True
        
        

    def search(self, word: str) -> bool:
        """
        Returns if the word is in the trie.
        """
        cur_trie = self.trie
        for c in word:
            if c not in cur_trie.subtries: 
                return False
            cur_trie = cur_trie.subtries[c]
        return cur_trie.is_word
            
        

    def startsWith(self, prefix: str) -> bool:
        """
        Returns if there is any word in the trie that starts with the given prefix.
        """
        cur_trie = self.trie
        for c in prefix:
            if c not in cur_trie.subtries: 
                return False
            cur_trie = cur_trie.subtries[c]
        return True 


# 648 
# application of trie
class Solution(object):
    def replaceWords(self, dic, sentence):
        """
        :type dict: List[str]
        :type sentence: str
        :rtype: str
        """
        trie = {}
        words = sentence.split(' ')
        self.build_trie(trie, dic)
        ans = []
        for w in words:
            flag = False
            for i in range(1, len(w)):
                if self.start_with(trie, w[:i]): 
                    flag = True
                    ans.append(w[:i])
                    break
            if not flag:
                ans.append(w)
        return ' '.join(ans)
    
    
    
    def build_trie(self, trie, words):
        for w in words:
            self.insert(trie, w)
    
    
    def insert(self, trie, word):
        for w in word:
            if w not in trie:
                trie[w] = {}
            trie = trie[w]
        trie['#'] = '#'
    
    
    def start_with(self, trie, prefix):
        for w in prefix:
            if w not in trie:
                return False
            trie = trie[w]
        if '#' not in trie: return False
        return True



# 211 
# backtracking + trie
class WordDictionary(object):

    def __init__(self):
        self.trie = {}
        
    def addWord(self, word):
        self.insert(word, self.trie)
        

    def search(self, word):
        return self.search_word(word, self.trie)
        
        
    def insert(self, word, trie):
        for w in word:
            if w not in trie:
                trie[w] = {}
            trie = trie[w]
        # ending condition, normal item is mapped to another map.
        trie['#'] = '#'
    

    def search_word(self, word, trie): 
        for i, w in enumerate(word):
            if w == '.':
                # may contain '#'
                for c in trie:
                    # do not include '#'
                    if c != '#': 
                        # be careful dont update trie outside, its backtracking. for example: trie = trie[c]
                        if self.search_word(word[i+1:], trie[c]):
                            return True
            if w not in trie: return False
            trie = trie[w]
        return '#' in trie
                
            

# 677
# implement partial trie using Node class
# tip be careful with the root referencing.
class TrieNode(object):
    def __init__(self):
        self.val = 0
#         [char, TrieNode]
        self.children = {}
        self.is_word = False    
    
    
class MapSum(object):

    def __init__(self):
        self.root = TrieNode()
        

    def insert(self, word, val):
        root = self.root
        for w in word:
            m = root.children
            if w not in m:
                m[w] = TrieNode()
            root = m[w]
        root.is_word = True 
        root.val = val
        
        
    def sum(self, prefix):
        cur = self.root
        for w in prefix:
            m = cur.children 
            if w not in m: 
                return 0 
            cur = m[w]
            ans = self.dfs(cur)
        return ans 
    
    
    def dfs(self, node):
        total = 0
        for child in node.children: 
            total += self.dfs(node.children[child])
        return total + node.val
        
    
  

#642
# trick: add [sentence: counts] mapping data to each node. thus each char will have a map of it sentence that contains it to counts
# for string a, b, when two strings can both reach some char k, meaning they have same prefix a[:k] == b[:k] before. 
class TrieNode(object):
    def __init__(self):
        self.children, self.counts = {}, {}
    
class AutocompleteSystem(object):
    def __init__(self, sentences, times):
        self.root, self.prefix = TrieNode(), ''
        for sentence, count in zip(sentences, times):
            self.insert(sentence, count)

            
    def insert(self, sentence, count):
        cur = self.root
        for c in sentence:
            if c not in cur.children:
                cur.children[c] = TrieNode()
            cur = cur.children[c]
            # if same sentence appear again, increment by 1, where '#' is encountered
            cur.counts[sentence] = cur.counts.get(sentence, 0) + count
        
        
    def input(self, c):
        if c == '#':
            self.insert(self.prefix, 1) # increment count of this sentence by 1
            self.prefix = ''
            return []
        
        self.prefix += c
        cur = self.root
        for c in self.prefix:
            if c not in cur.children: return []
            cur = cur.children[c]
        
        hp = []
        for s in cur.counts:
            heapq.heappush(hp, (-cur.counts[s],s))

        ans, i = [], 3
        while hp and i > 0:
            _ , c = heapq.heappop(hp)
            ans.append(c)
            i -= 1
        return ans 
            



# 588
# tip: trie is not necessarily used to store characters, it can use to store token, like this problem it stores directories.
# this is good abstraction and application of trie 
class File(object):
    def __init__(self):
        self.is_file = False
        self.children = {} # store dir -> file nodes mappping
        self.content = ''


class FileSystem(object):

    def __init__(self):
        self.root = File()

        
    def ls(self, path):
        """
        :type path: str
        :rtype: List[str]
        """
        res = []
        cur = self.root
        for d in path.split('/'):
            if d == '': continue 
            if d not in cur.children: 
                return res
            cur = cur.children[d]
        
        if cur.is_file:
            res.append(d)
            return res
        
        for f in cur.children:
            res.append(f)
        
        return sorted(res)
        
        
    def mkdir(self, path):
        """
        :type path: str
        :rtype: None
        """
        cur = self.root
        for d in path.split('/'):
            if d == '': continue 
            if d not in cur.children:
                cur.children[d] = File()
            cur = cur.children[d]
         

    def addContentToFile(self, filePath, content):
        """
        :type filePath: str
        :type content: str
        :rtype: None
        """
        cur = self.root
        for d in filePath.split('/'):
            if d == '': continue 
            if d not in cur.children:
                cur.children[d] = File()
            cur = cur.children[d]
        cur.is_file = True
        cur.content += content
                
        

    def readContentFromFile(self, filePath):
        """
        :type filePath: str
        :rtype: str
        """
        cur = self.root
        for d in filePath.split('/'):
            if d == '': continue 
            if d not in cur.children:
                cur.children[d] = File()
            cur = cur.children[d]
        return cur.content
        
        

# 297 
# Definition for a binary tree node.
class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

class Codec:

    def serialize(self, root):
        """Encodes a tree to a single string.
        
        :type root: TreeNode
        :rtype: str
        """
        res = []
        self.pre_order_serialize(root, res)
        return ','.join(res)
    
    
    def pre_order_serialize(self, root, res):
        if not root: 
            res.append('#')
            return 
        
        res.append(str(root.val))
        self.pre_order_serialize(root.left, res)
        self.pre_order_serialize(root.right, res)
        
            
        
    def deserialize(self, data):
        """Decodes your encoded data to tree.
        
        :type data: str
        :rtype: TreeNode
        """
        # use comma as delimiter coz the value may be negative or more than 1 digit
        return self.build_tree(data.split(','), 0)[1]
    
    # trick is to update the index of current node.
    def build_tree(self, data, i): 
        if data[i] != '#':
            root = TreeNode(int(data[i]))
            j, left = self.build_tree(data, i+1)
            j, right = self.build_tree(data, j+1)
            i = j
            root.left = left
            root.right = right
            return i, root
        return i, None
        
# implement DFS and BFS solutions



# 295 
class MedianFinder(object):

    def __init__(self):
        self.min_heap = []
        self.max_heap = []
        

    # trick: make the min_heap to store larger numbers, the larger number heap should at least the same size of smaller number heap. 
    # when a uew num comming in, first to compare with smallest in min heap, and pop the smallest off and add it to the max_heap
    # and at the same time maintain the property that large num heap >= small num heap.
    def addNum(self, num):
        heapq.heappush(self.min_heap, num)
        mi = heapq.heappop(self.min_heap)
        heapq.heappush(self.max_heap, -mi)
        if len(self.min_heap) < len(self.max_heap):
            mx = heapq.heappop(self.max_heap)
            heapq.heappush(self.min_heap, -mx)
            
            
    def findMedian(self):
        # if odd
        if len(self.min_heap) > len(self.max_heap):
            return float(self.min_heap[0])
        # if even
        return float(self.min_heap[0]-self.max_heap[0])/2

        
""" Binary Search """

""" 统一使用左闭右闭模版  [l, r] or [l,r)  """
# left <= right
# 有些题目 需要使用 left + 1 < right
# shrinking the left and right boundries until left > right meaning nothing found or left is the inserting point
# tip:
# 1. understand what are you searching for first
# 2. think about what condition to only consider half of the given array.
# 3. follow the 模版 and be consistent.
# 4. left and right (inclusive) meaning the possible range that target is located within.
""" 统一使用左闭右闭模版  [l, r] or [l,r)  """

# p35
# [l, r]
# find the index of the first number that is greater than or equal to the target
# 因为右闭， mid会偏向左边小的那个数， left <= right 这个条件 会多走一个循环 导致left 会落在大的那个数 也就是要插入的点
def searchInsert(nums, target):
    left, right = 0, len(nums)-1
    while left <= right:
        mid = (left + right)//2
        if nums[mid] == target:
            return mid
        if nums[mid] > target:
            right = mid - 1
        else:
            left = mid + 1
    # because the ending conditon is left > right. so left is the place to insert
    return left

# [l, r)
# 因为右开， mid会偏向右边, 也就是偏向大的那个数, 也就是最后我们要找的或者插入的地方
def searchInsert_(self, nums, target):
    left, right = 0, len(nums)
    
    while left < right:
        mid = (left+right)//2
        if nums[mid] < target:
            left = mid + 1
        else:
            right = mid
    
    return left

# 不建议用这个模版
# [l, r] and left + 1 < right
# 左右夹击 左右每次只移动到 mid， 能保证最后会有两个可能的 candidates， 然后逐一验证 
def searchInsert(self, nums, target):
    left, right = 0, len(nums)-1
    
    while left + 1 < right:
        mid = (left+right)//2
        if nums[mid] < target:
            left = mid
        else:
            right = mid
            
    if nums[left] >= target: return left
    if nums[right] >= target: return right
    if nums[right] < target: return right+1



# p34
# [l, r] O(n)
def searchRange(self, nums, target):
    res = [-1, -1]
    
    if not nums: return res
    
    def bsearch(nums, target):
        left, right = 0, len(nums)-1
        while left <= right: 
            mid = (left+right)//2
            if nums[mid] == target: return mid 
            elif nums[mid] > target: right = mid -1 
            else: left = mid + 1
        return -1
    
    pos = bsearch(nums, target)

    if pos == -1: return res

    l, r = pos-1, pos+1 

    while l >= 0 and nums[l] == nums[pos]: l -= 1
    res[0] = l + 1
    
    while r < len(nums) and nums[r] == nums[pos]: r += 1
    res[1] = r - 1 
    
    return res 


# [l,r) lgn 
def searchRange(self, nums, target):
    res = [-1, -1]
    if not nums: return res

    left, right = 0, len(nums)
    while left < right:
        mid = (left + right)//2
        # shrink the window to closer to target 
        if nums[mid] < target:
            left = mid + 1 
        else:
            right = mid
    
    if 0 <= left < len(nums) and nums[left] == target:
        res[0] = left

    """find the first number that is > target is the way to find the right most target"""
    left, right = 0, len(nums)
    while left < right:
        mid = (left + right)//2
        # opposite direction, to find greater than target number.
        if nums[mid] <= target:
            left = mid + 1
        else:
            right = mid
            
            
    if nums[left-1] == target:
        res[1] = left-1
    
    return res


# optimal solution, same idea but more concise
def searchRange(self, nums, target):
    res = [-1, -1]
    if not nums: return res

    def bsearch(nums, target):
        left, right = 0, len(nums)
        while left < right:
            mid = (left + right)//2

            if nums[mid] < target:
                left = mid + 1 
            else:
                right = mid
        return left 
    
    # find the left most target
    left = bsearch(nums, target)
    # find the left most position for target+1 to be inserted, thus -1 will be the right most target
    right = bsearch(nums, target+1)-1
    
    if 0<= left < len(nums) and nums[left] == target: 
        return [left, right]
    
    return res 
        



# mix of two templates, 不推荐
def searchRange(self, nums, target):
    def search(nums, target):
        left, right = 0, len(nums)-1
        while left <= right:
            mid = (left + right)//2
            if nums[mid] < target:
                left = mid + 1
            else:
                right = mid - 1
        return left

    ans = [-1, -1]
    if nums:
        # find the first element that is <= target
        start = search(nums, target)
        # find the first element taht is greater than target. this returns a insertion point.
        end = search(nums, target + 1) - 1
        if 0 <= start < len(nums) and nums[start] == target:
            ans[0], ans[1] = start, end
    return ans




#p33
# tip: consider target and mid on the same side and not on the same side, updating left and right pointer accordingly
def search(self, nums, target):
    if not nums:
        return -1
    left, right = 0, len(nums)-1
    while left <= right:
        mid = (left + right) >> 1
        if nums[mid] == target:
            return mid
#           mid is on the Sorted array
        if nums[mid] < nums[-1]:
            if nums[mid] < target <= nums[-1]:
                left = mid + 1
            else:
                #                   target is on the rotated sorted array
                right = mid - 1
        else:
            #                 mid is on rotated sorted array
            if nums[left] <= target < nums[mid]:
                right = mid - 1
            else:
                #                  target is on the sorted array
                left = mid + 1
    return -1



# p153
# [l, r)
def findMin(self, nums):
    left, right = 0, len(nums)
    while left < right:
        mid = (left + right)//2
        
        if nums[mid] <= nums[-1]:
            right = mid
        else:
            left = mid + 1
    return nums[right]


# 154



# p162
# there are some parts of the array is increasing or decreasing st is sorted. thus this is a hint to use binary search.
# four cases to handle: mid is peak, mid is on increasing sequence, mid is on decreasing sequence, mid is valley. 
def findPeakElement(self, nums):
    if len(nums) == 1:
        return 0
    left, right = 0, len(nums)-1
    # if using left <= right condition, better think about if left == right, what should we return.
    while left <= right:
        mid = (left + right)//2
        if (mid - 1 < 0 or nums[mid-1] < nums[mid]) and (mid+1 >= len(nums) or nums[mid] > nums[mid+1]):
            return mid
        # moving left and right boundry not necessarily need to plus 1 or minus one
        if nums[mid] < nums[mid+1]:
            left = mid+1
        else:
            right = mid


# [l, r]
def findPeakElement(self, nums):
    nums =[float('-inf')] +  nums + [float('-inf')]
    left, right = 1, len(nums)-2
    while left <= right:
        mid = (left + right)//2  
        
        if nums[mid-1] < nums[mid] and nums[mid] > nums[mid+1]: 
            return mid-1
        elif nums[mid-1] > nums[mid]:
            right = mid - 1
        else:
            left = mid + 1
    return -1


# mono stack 
def findPeakElement(self, nums):
    stack = []
    for i in range(len(nums)+1):
        n = nums[i] if i < len(nums) else float('-inf')
        if stack and nums[stack[-1]] > n:
            return stack.pop()
        stack.append(i) 
    return -1



# p852
def peakIndexInMountainArray(self, A):
    if len(A) >= 3:
        left, right = 0, len(A)-1
        while left <= right:
            mid = (left + right)//2
            if mid-1 >= 0 and mid+1 < len(A) and A[mid] > A[mid-1] and A[mid] > A[mid+1]:
                return mid
            if A[mid] < A[mid+1]:
                left = mid + 1
            else:
                right = mid
    return -1



# p69
def mySqrt(self, x):
    left, right = 1, x
    while left <= right:
        mid = (left + right)//2
        if mid*mid == x:
            return mid
        if mid*mid < x:
            left = mid + 1
        else:
            right = mid - 1
    return right


# p74
def searchMatrix(self, matrix, target):
    if not matrix:
        return False
    left, right = 0, len(matrix)*len(matrix[0])-1
    while left <= right:
        mid = (left + right)//2
        row, col = divmod(mid, len(matrix[0]))
        if matrix[row][col] == target:
            return True
        if matrix[row][col] < target:
            left = mid + 1
        else:
            right = mid - 1
    return False



# 81
# hard
def search(self, nums, target):
    if not nums:
        return False
    left, right = 0, len(nums)-1
    while left <= right:
        mid = (left + right) >> 1
        if nums[mid] == target:
            return True

        while left < mid and nums[mid] == nums[left]:
            left += 1
        
        if nums[mid] >= nums[left]:
            if nums[left] <= target <= nums[mid]:
                right = mid -1 
            else:
                left = mid + 1
        else:
            if nums[mid] < target <= nums[right]:
                left = mid + 1
            else:
                right = mid - 1
    return False



# p875
# search for right speed. inituition using binary search
def minEatingSpeed(self, piles: List[int], H: int) -> int:
    import math
    # search space. 
    left, right = 1, max(piles)
    while left < right: 
        mid = (left + right) //2 
        # mid too small
        if sum(math.ceil(p/mid) for p in piles) > H:
            left = mid + 1
        else:
            right = mid 
    return left



# p378
def kthSmallest(self, matrix: List[List[int]], k: int) -> int:
    left, right = matrix[0][0], matrix[-1][-1]  
    
    while left < right:
        mid = (left + right)//2
        j = len(matrix[0])-1
        cnt = 0 
        # j is tricky here, the numbers that my <= mid is located at (<= i, <= j), everything after (>i, >j) will be greater than mid
        for r in matrix:
            while j >= 0 and r[j] > mid: j -= 1
            cnt += (j+1)
        
        if cnt < k: 
            left = mid + 1
        else:
            right = mid 
    return left



# 1011
def shipWithinDays(self, weights: List[int], D: int) -> int:
    left, right = max(weights), sum(weights)
    
    while left < right:
        mid, d, cur = (left+right)//2, 1, 0 
        for w in weights:
            if w + cur > mid:
                d += 1
                cur = 0
            cur += w
        if d > D:
            left = mid + 1
        else:
            right = mid 
    return left


# 287
# trick: use the pigeonhole princicple. 
# keys: 
# 1. only has one duplicate in the array. 
# 2. search space 1 to n, meaning we have n buckets to put n+1 indices. the duplicate must be one side of the 1 to n 
def findDuplicate(self, nums):
    left = 1
    right = len(nums)-1
    while left < right:
        mid = (left+right)//2
        cnt = 0
        # find out how many items put into the buckets on the mid or to the left of mid 
        for n in nums:
            if n <= mid:
                cnt += 1
        # the duplicate is on the other half of the buckets
        if cnt <= mid:
            left = mid + 1
        else:
            right = mid   
    return left
                

# trick: toitois and hare principle, this is the same as the linkedlist cycle detection and where cycle entrance is problem. 
def findDuplicate(self, nums):
        slow = nums[0]
        fast = nums[nums[0]]
        
        while slow != fast:
            slow = nums[slow]
            fast = nums[nums[fast]]
        
        fast = 0 
        while slow != fast:
            fast = nums[fast]
            slow = nums[slow]
        return fast



""" dfs/bfs/backtrack/tree/graph/uf """
# 987
# key: heappop() function will perform normal pop [0] item off the array, then turn the array into a heap. this function assume the given array 
# is already a heap. so before use this function, call heapify() first or use heappush first.
def verticalTraversal(self, root: TreeNode) -> List[List[int]]:
    dic = collections.defaultdict(list)
    self.visit(root, 0, 0, dic)
    res = []
    for k in sorted(dic.keys()):
        size = len(dic[k])
        # so the same x coord, the higher level node value comes first.
        heapq.heapify(dic[k])
        tmp = [heapq.heappop(dic[k])[1] for _ in range(size)] 
        res += [tmp]
    return res


def visit(self, root, x, level, dic):
    if not root: return
    dic[x].append((level, root.val))
    self.visit(root.left, x-1, level+1, dic)
    self.visit(root.right, x+1, level+1, dic)

# 261 
# dfs + cycle detection 
# check cycle, if no cycle, check size of visited nodes, if equal to n, true, else false
def validTree(self, n: int, edges: List[List[int]]) -> bool:
    # build the undirected graph
    graph = collections.defaultdict(list)
    for u, w in edges:
        graph[u].append(w)
        graph[w].append(u)
    
    visited = [None] * n
    if self.hasCycle(0, graph, visited, 0): 
        return False
    
    for v in visited:
        if not v: return False
    return True


# tip:  A-B for undirected graph forms a cycle. for directed graph, this will not form a cycle. thus we use a pre to avoid this case.
# meaning dont visted the path you just came from.
def hasCycle(self, start, graph, visited, pre):
    if visited[start]: 
        return visited[start] == 'yes'
    visited[start] = "yes"
    for nei in graph[start]:
        if nei == pre: continue
        if self.hasCycle(nei, graph, visited, start):
            return True
    visited[start] = 'no'
    return False



# 323
# do dfs on each node and mark visited, when visited == n, no more component exists
def countComponents(self, n: int, edges: List[List[int]]) -> int:
    visited = set()
    # build graph
    graph = collections.defaultdict(list)
    for u, w in edges:
        graph[u].append(w)
        graph[w].append(u)
    
    cnt = 0 
    for i in range(n):
        if len(visited) == n: break
        if i not in visited:
            self.dfs(i, graph, visited)
            cnt += 1
    return cnt


def dfs(self, start, graph, visited):
    visited.add(start)
    for nei in graph[start]:
        if nei not in visited:
            self.dfs(nei, graph, visited)
        


# 684
# Check cycle: for edge[u, v], check if node u can reach v in other ways beside the edge itself.
# key: the first edge that closes the cycle is the last edge that in that cycle. 
def findRedundantConnection(self, edges: List[List[int]]) -> List[int]:
    #check cycle as you go building the graph
    graph = collections.defaultdict(list)
    for u, v in edges:
        if self.canReach(0, u, v, graph):
            return [u, v]
        else:
            graph[u].append(v)
            graph[v].append(u)
    return []


# check if start reaches end, if does, has cycle else no cycle
# pre is used to avoid going backward
def canReach(self, pre, start, end, graph):
    if start == end: 
        return True 
    
    for w in graph[start]:
        if w == pre: continue
        if self.canReach(start, w, end, graph):
            return True
    return False
    
# union find implementation.  


# 1043
# dfs: max sum of given array after partitioning 
# choices: [start : start+i] where i in the range of [:k]
# max(max([start : start+i]) * (i+1) + dfs())
def maxSumAfterPartitioning(self, A: List[int], K: int) -> int:
    return self.dfs(A, 0, K, {})


# place par at i where i in [start: start+k], do the same for the arr [start+i+1: ]
def dfs(self, nums, start, k, seen):
    if start == len(nums): return 0
    if start in seen: return seen[start]
    max_sum = 0 
    
    end = start + k if start + k <= len(nums) else len(nums)
    for i in range(start, end):
        max_sum = max(max_sum, self.dfs(nums, i+1, k, seen) + max(nums[start: i+1]) * (i-start+1))
    seen[start] = max_sum
                
    return max_sum
    
    


# 1042
def gardenNoAdj(self, N: int, paths: List[List[int]]) -> List[int]:
        graph = collections.defaultdict(list)
        for a, b in paths:
            graph[a-1].append(b-1)
            graph[b-1].append(a-1)
        path = [0]*N 
        self.placeFlowers(0, graph, path, N)
        return path
    
# pick a flower and place it into current garden if this flower was not place at neibhors, do the same for the rest of gardens.
def placeFlowers(self, cur, graph, path, N):
    if cur == N: return True
    for color in [1,2,3,4]:
        same = False
        for nei in graph[cur]:
            if path[nei] == color:
                same = True 
                break
        if same: continue
        path[cur] = color
        if self.placeFlowers(cur+1, graph, path, N): return True
        path[cur] = 0 
    return False



# 254
# backtrack
# choices: [i: n/i] where i in [2:n]
# terminate: target <= 1 and path length > 1
# T:O(), S:O()
# key: notice that if you pick i, the possible candidates will be within n/i. this will largely reduce the unnecesary work.
# tip: unnecessary work usually occurs at there are too many unnecessary choices for backtrack/dfs problems.
def getFactors(self, n: int) -> List[List[int]]:
    res = []
    self.backtrack(2, [], res, n)
    return res
    
    
def backtrack(self, start, path, res, target):
    if target <= 1:
        if len(path) > 1:
            res.append(path)
        return

    for i in range(start, target+1):
        if target % i == 0: 
            self.backtrack(i, path + [i], res,  target // i)



# 277
# greedy.
# A knows B, A is not candidate of celebrity. 
# A does not know B, B is not candidate of celebrity
# T: O(n), S:O
def findCelebrity(self, n: int) -> int:
    cand = 0 
    for i in range(1, n):
        if knows(cand, i):
            cand = i
    
    for j in range(n):
        if cand != j and (not knows(j, cand) or knows(cand, j)):
            return -1
        
    return cand



# 1129
# bfs
# intuition: shortest path -> bfs or dikstra 
# state: (node, color)
# start: [(0, red), (0, blue)]
# data structure: queue, hashtable
def shortestAlternatingPaths(self, n: int, red_edges: List[List[int]], blue_edges: List[List[int]]) -> List[int]:
    queue = collections.deque([(0, 0), (0, 1)])
    red, blue = collections.defaultdict(list), collections.defaultdict(list)
    for s, e in red_edges: red[s].append(e)
    for s, e in blue_edges: blue[s].append(e)
    ans = [-1] * n
    seen = set()
    dist = 0
    while queue:
        size = len(queue)
        for _ in range(size):
            node, color = queue.popleft()
            # only when the node is not assign with a value before
            if ans[node] == -1:
                ans[node] = dist
            if (node, color) not in seen:
                seen.add((node, color))
                if color == 0: 
                    if node in blue:
                        for nd in blue[node]:
                            queue.append((nd, 1))
                else:
                    if node in red:
                        for nd in red[node]:
                            queue.append((nd, 0))
        dist += 1
    return ans
            
            

# 490 
def hasPath(self, maze: List[List[int]], start: List[int], destination: List[int]) -> bool:
    return self.canReach(start, destination, set(), maze)

# 目标是寻找路径，dfs 深度 查询 每次移动到下一个位置， 如果当前位置就是终点 返回True， 寻找路径问题 如果当前访问的位置已经出现过， 那么说明我们遇到了环
# 环说明当前路径永远没办法到达终点 
# 寻找路径问题的 False base条件：遇环
def canReach(self, cur, dest, visited, maze):
    if cur == dest: return True
    i, j = cur
    # 遇到环
    if (i, j) in visited:
        return False;

    visited.add((i, j))
    for x, y in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
        a, b = i,j
        while 0 <= a + x < len(maze) and 0 <= b + y < len(maze[0]) and maze[a+x][b+y] != 1:
            # 这里很容易出错 容易写成 x += b, 是要增加一个步
            a += x
            b += y
        if self.canReach([a, b], dest, visited, maze):
            return True
    return False


# bfs
# O(n*m) worse case the entire maze traversed to find the path
# space O(n*m)
def canReach_bfs(self, maze, start, dest, visited):
    dir = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    queue = collections.deque([start])
    while queue:
        size = len(queue)
        while size:
            x, y = queue.popleft()
            if [x, y] == dest: return True
            visited.add((x, y))
            for i, j in dir:
                nx, ny = x, y
                while 0 <= nx + i < len(maze) and 0 <= ny + j < len(maze[0]) and maze[nx+i][ny+j] != 1:
                    nx += i
                    ny += j
                if (nx,ny) not in visited:
                    queue.append((nx, ny))
            size -= 1
    return False
        
        
        

# 505
# dykstra algorithm.
# tip: parent to each neighbor has weights/distance, is not same anymore, so we need to use priorityqueue to put shortest path so far on the top of heap.
def shortestDistance(self, maze, start, destination):
    queue = []
    heapq.heappush(queue, (0, start[0], start[1]))
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    
    while queue:
        dist, i, j = heapq.heappop(queue)
        if i == destination[0] and j == destination[1]: return dist
        maze[i][j] = 2

        for x, y in directions:
            row = i + x 
            col = j + y
            d = 0
            while 0 <= row < len(maze) and 0 <= col < len(maze[0]) and maze[row][col] != 1:
                row += x 
                col += y
                d += 1
            row -= x 
            col -= y
            if maze[row][col] == 0: 
                heapq.heappush(queue, (dist+d, row, col))
    return -1


# 286
# 逆向思考， 收集所有为0的位置 存（i，j，0）进queue， 然后同时向四个方向扩张 如果遇到inf 就把距离加1 然后把这个位置加进queue里
# bfs
def wallsAndGates(self, rooms: List[List[int]]) -> None:
        if not rooms: return 
        n, m = len(rooms), len(rooms[0])
        queue = collections.deque()
        for i in range(n):
            for j in range(m):
                if rooms[i][j] == 0:
                    queue.append((i,j, 0))
        self.bfs(queue, rooms)
        
#  you dont need the visited set, because we only append the positions that has inf as value, but during expanding, we have changed
#  the position having inf with the shortest distance to gate, so this position will not cause problem. 
def bfs(self, queue, rooms):
    visited = set()
    d = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    while queue:
        size = len(queue)
        while size:
            x, y, z = queue.popleft()
            visited.add((x, y))
            for i, j in d:
                i += x
                j += y
                if 0 <= i < len(rooms) and 0 <= j < len(rooms[0]) and (i, j) not in visited:
                    if rooms[i][j] == -1: continue
                    if rooms[i][j] == 2147483647:
                        rooms[i][j] = z+1
                        queue.append((i, j, z+1))
            size -= 1
            
# dfs
def wallsAndGates(self, rooms: List[List[int]]) -> None:
        if not rooms: return 
        n, m = len(rooms), len(rooms[0])
        for i in range(n):
            for j in range(m):
                if rooms[i][j] == 0:
                    self.dfs(rooms, i, j , 0)
        
#  d > rooms[i][j]: if the cell is updated before and the distance is is less than d, no need to continue from current position
# updated shortest distance to current gate for all inf.  
def dfs(self, rooms, i, j, d): 
    if i < 0 or i >= len(rooms) or j < 0 or j >= len(rooms[0]) or d > rooms[i][j]: 
        return
    rooms[i][j] = d
    for x, y in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        self.dfs(rooms, i+x, j+y, d+1)
        
        
        
# 542 similar to above problem
# dfs
def updateMatrix(self, matrix: List[List[int]]) -> List[List[int]]:
    if not matrix: return 
    n, m = len(matrix), len(matrix[0])
    for i in range(n):
        for j in range(m):
            if matrix[i][j] == 1:
                matrix[i][j] = float('inf')
    
    for i in range(n):
        for j in range(m):
            if matrix[i][j] == 0:
                self.dfs(matrix, i, j, 0)
    return matrix


def dfs(self, matrix, i, j, d):
    if i < 0 or i >= len(matrix) or j < 0 or j >= len(matrix[0]) or matrix[i][j] < d:
        return
    
    matrix[i][j] = d
    for x, y in [(0,1), (0, -1), (1, 0), (-1, 0)]:
        self.dfs(matrix, i+x, j+y, d+1)

        
#  bfs
def updateMatrix(self, matrix):
    n, m = len(matrix), len(matrix[0])
    import collections
    queue = collections.deque()
    for i in range(n):
        for j in range(m):
            if matrix[i][j] == 0:
                queue.append((i, j))
            else:
                matrix[i][j] = float('inf')

    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    while queue:
        i, j = queue.popleft()
        for x, y in directions:
            r = i + x
            c = j + y
            z = matrix[i][j] + 1

            if r < 0 or r >= n or c < 0 or c >= m or matrix[r][c] <= z:
                    continue

            queue.append((r, c))
            matrix[r][c] = z
    return matrix




# 1130
# top down memo
# time O(n^2) space O(n^2)
def mctFromLeafValues(self, arr):
    return self.dfs(0, len(arr)-1, arr, {})

# separate into left subtree leaves and right subtree leaves. 
def dfs(self, start, end, arr, visited):
    if start == end: return 0 
    if (start, end) in visited: return visited[(start, end)]
    
    mi = float('inf')
    # end not end + 1 coz the tree is strict binary tree. either 0 or 2 children.
    for i in range(start, end):
        mi = min(mi, self.dfs(start, i, arr, visited) + self.dfs(i+1, end, arr, visited) + max(arr[start:i+1])*max(arr[i+1:end+1]))
    visited[(start, end)] = mi
    return mi 


# dp
def mctFromLeafValues(self, arr: List[int]) -> int:
    dp = [[float('inf')] * len(arr) for _ in range(len(arr))]
    for i in range(len(arr)-1, -1, -1):
        dp[i][i] = 0
        for j in range(i, len(arr)):
            for k in range(i, j):
                dp[i][j] = min(dp[i][j], dp[i][k]+dp[k+1][j]+ max(arr[i:k+1]) * max(arr[k+1:j+1]))
    return dp[0][len(arr)-1]


# greedy
# time O(n^2) space O(n)
# trick: abstract the problem to: for two leaves left, right, in order to get rid of min(left, right), it will cost left * right. 
# at the end only one node is left. thus we greedily pick the smallest node to be gotten rid of. 
def mctFromLeafValues_greed(self, A):
    res = 0
    while len(A)>1: 
        i = A.index(min(A))
        res += min(A[i-1:i]+A[i+1:i+2])*A.pop(i)
    return res


# use mono stack to find the valley and do the removing operations
# time O(n) space O(n)
def mctFromLeafValues(self, A):
    res = 0
    stack = [float('inf')]
    for a in A:
        while stack[-1] <= a:
            v = stack.pop()
            res += v*min(stack[-1], a)
        stack.append(a)
    #decreasing stack, just get rid of rightmost element first
    while len(stack) > 2:
        res += stack.pop() * stack[-1]
    return res 

    
        

# 394
def decodeString(self, s):
    return ''.join(self.decode(s))


def decode(self, s):
    if not s: return []
    ans = []
    if not s[0].isdigit():
        ans += [s[0]] + self.decode(s[1:])
    else:
        j = 0
        while s[j].isdigit(): j += 1
        n = int(s[:j])
        j += 1
        begin, bal = j, 1
        while j < len(s) and bal != 0:
            if s[j] == '[': bal += 1
            if s[j] == ']': bal -= 1
            j += 1
        ans += n * self.decode(s[begin:j-1]) + self.decode(s[j:])
    return ans
  
        
# stack
def decodeString(self, s: str) -> str:
    str_stack, num_stack = [], []
    cur_str = ''
    cur_num = 0
    for c in s:
        if c == '[':
            str_stack.append(cur_str)
            num_stack.append(cur_num)
            cur_str = ''
            cur_num = 0 
        elif c == ']':
            prev_str = str_stack.pop()
            num = num_stack.pop()
            cur_str = prev_str + num * cur_str
        elif c.isdigit():
            cur_num = cur_num * 10 + int(c)
        elif c.isalpha():
            cur_str += c
    return cur_str


# 994
# trick: need to push all the rotton oranges to the queue at once
def orangesRotting(self, grid):
    n, m = len(grid), len(grid[0])
    visited = set()
    mi = self.bfs(grid, visited)
    return mi


def bfs(self, grid, visited):
    n, m = len(grid), len(grid[0])
    queue = collections.deque()
    cnt = 0
    for i in range(n):
        for j in range(m):
            if grid[i][j] == 1:
                cnt += 1
            if grid[i][j] == 2:
                visited.add((i,j))
                queue.append((i,j))
                
    m = 0 
    while queue:
        size = len(queue)
        cur = cnt
        while size:
            size -= 1
            a, b = queue.popleft()
            directions = [(0,1), (0,-1), (1, 0), (-1, 0)]
            for x, y in directions:
                x += a
                y += b
                if 0 <= x < len(grid) and 0 <= y < len(grid[x]) and (x, y) not in visited and grid[x][y] != 0:
                    visited.add((x, y))
                    if grid[x][y] == 1:
                        cnt -= 1
                        grid[x][y] = 2
                    queue.append((x, y))
        if cur > cnt: m += 1
    return -1 if cnt != 0 else m
       
            
            


# 17
# tip: because each digit, you need to pick one and can only pick one, cannot pick multiple chars from one digit. thus, when you pick one char out of one digit
# you need to move on to next digit. so we use an index to mark current picked digit. when the choice is fixed or only one choice, you can use stack frame and index to
# track it.
def letterCombinations(self, digits):
    chars = {'2':'abc', '3':'def', '4':'ghi', '5':'jkl', '6':'mno', '7':'pqrs', '8':'tuv', '9':'wxyz'}
    res = []
    if not digits: return res
    self.dfs(0, digits, chars, '', res)
    return res

# place one of the candidates char into len(digits) 
# total len(digits) spot, idx is currently working on spot, select one char from [idx] and do the same for the rest spot
def dfs(self, idx, digits, chars, path, res):
    if idx == len(digits):
        res.append(path)
        return
    
    for c in chars[digits[idx]]:
        self.dfs(idx+1, digits, chars, path+c, res)



# iterative solution: bfs
def helper_iter(self, mapping, digits):
    queue = collections.deque()
    queue.append('')
    cur = 0  # index of current processing digit
    while queue and cur < len(digits):
        size = len(queue)
        while size:
            front = queue.popleft()
            # append front in front of all the neighbors and add to the queue
            for c in mapping[digits[cur]]:
                queue.append(front + c)
            size -= 1
        cur += 1
    return list(queue)




# 39
# backtrack
# choices: candidates
# with sort and break early improve the efficiency
# possible result order does not matter, so [1, 2] is the same as [2,1]
# T: O(N*2^N), S:O(N*2^N)
def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
    res = [] 
    candidates.sort()
    self.backtrack(0, candidates, [], res, target)
    return res

# must use "start" to avoid repeats like [1,2] and [2,1].
def backtrack(self, start, candidates, path, res, target):
    if target == 0:
        res.append(path)
        return 
    
    for i in range(start, len(candidates)):
        if candidates[i] > target: break
        self.backtrack(i, candidates, path + [candidates[i]], res, target - candidates[i])
        


""" dp solution"""
# dp[t]: combinations that sum to t
# dp[t] : dp[t - cand[j]]


def combinationSum(self, candidates, target):
    candidates.sort()
    res = self.helper(candidates, target)
    return res[target]


def helper(self, cand, tar):
    dp = [[] for _ in range(tar+1)]
    # for every t:
    for t in range(tar+1):
        tmp = []
        j = 0
        while j < len(cand) and cand[j] <= t:
            if cand[j] == t:
                tmp.append([cand[j]])
            else:
                for comb in dp[t-cand[j]]:
                    # for every cand[j] < t, we start from j == 0, so there could be duplicates
                    # to avoid duplicates [2,2,3] [2,3,2], we mantain the combination list in ascending order
                    if cand[j] >= comb[-1]:
                        tmp.append(comb + [cand[j]])
            j += 1
        dp[t] = tmp
    return dp



# 40
# backtrack
# with dup, so sort the array, so that dup can be skipped
# combination, so j > i, comb start with [j] cannot go back to i
# T: O(2^N) same as get all subsets, S:O(N * 2^N)
def combinationSum2(self, candidates, target):
    def find_comb(start, path, rem):
        if rem == 0:
            res.append(path)
            return
        for i in range(start, len(candidates)):
            """ be careful with the i-1>= start condition, don't code it like i-1>= 0
            otherwise you will skip some case such as 1,1,6
            on the other hand, if there are duplicates, we need to skip the latter ones
                """
            if i-1 >= start and candidates[i] == candidates[i-1]:
                continue
            if rem - candidates[i] < 0:
                break
            find_comb(i+1, path+[candidates[i]], rem - candidates[i])

    res = []
    candidates.sort()
    find_comb(0, [], target)
    return res



def combinationSum2(self, candidates, target):
    candidates.sort()
    dp = [set() for i in range(target+1)]
    dp[0].add(())
    for num in candidates:
        for t in range(target, num-1, -1):
            for prev in dp[t-num]:
                dp[t].add(prev + (num,))
    return list(dp[-1])




# 77
# backtrack
# combination dose not care about order, permutation care about order.
# T:O(), S:O()
def combine(self, n, k):
    def build_comb(start, path, rem):
        if rem == 0:
            res.append(path)
            return
        for i in range(start, n+1):
            build_comb(i+1, path+[i], rem-1)

    res = []
    build_comb(1, [], k)
    return res


""" iterative using pointer """
# making all the elements in the path increment by 1 initially,and then working backward from the position k-1 backed to 0

def combine_iter(self, n, k):
    res, path = [], [0]*k
    i = 0
    while i >= 0:
        path[i] += 1
        # moving backward if current position is exceeding the n
        if path[i] > n:
            i -= 1
        # add the result
        elif i == k-1:
            res.append(path[:])
        # moving forward if path[i] <= n and i < k-1
        else:
            i += 1
            # this line and the "path[i] += 1" line can guruntee [i] > [i-1]
            path[i] = path[i-1]
    return res


""" iterative dfs """
def combine_iter(self, n, k):
    ans, stack, x = [], [], 1
    while stack or x <= n:
        while len(stack) < k and x <= n:
            stack.append(x)
            x += 1

        if len(stack) == k:
            ans.append(stack[:])

        x = stack.pop() + 1
    return ans




def combine_binary(self, n, k):
    # init first combination
    nums = list(range(1, k + 1)) + [n + 1]

    output, j = [], 0
    while j < k:
        # add current combination
        print(nums)
        output.append(nums[:k])
        # increase first nums[j] by one
        # if nums[j] + 1 != nums[j + 1]
        j = 0
        while j < k and nums[j + 1] == nums[j] + 1:
            nums[j] = j + 1
            j += 1
        nums[j] += 1

    return output



""" implement math solution """
# formula C(n, k) = C(n-1, k-1) + C(n-1, k), very easy to prove
# means if n is selected, go select k-1 from n-1, elif n is not selectd, go select k from n-1


def combine_math(self, n, k):
        # C(n-1, k)
        if n == k:
            return [[i for i in range(1, n+1)]]
        # C(n-1, k-1)
        if k == 1:
            return [[i] for i in range(1, n+1)]

        return self.combine(n-1, k) + [[n] + j for j in self.combine(n-1, k-1)]


""" implement dp solution """
# C(n, k) = C(n-1, k-1) + C(n-1, k)


def combine_dp(self, n, k):
    C = [[[] for _ in range(k+1)] for _ in range(n+1)]
    for r in range(n+1):
        C[r][0] = [[]]

    for i in range(1, n+1):
        for j in range(1, k+1):
            if j > i:
                break
            if i-1 >= j:
                C[i][j] += C[i-1][j]
            C[i][j] += [c + [i] for c in C[i-1][j-1]]

    return C[n][k]


# 78
# backtrack
# add current set to the result, pick one num from nums append to the current set
# exit: until all options run out.
# T: O(n*2^n), S: O(N*2^N) there are 2^n power sets, each cost O(n)
# B(n) = B(n-1) + B(n-2) .. B(1)
# B(n+1) = B(n) + B(n-1) + B(n-2) .. B(1)
# B(n+1) = 2B(n)
# O(2^n) * time to copy array operations cost O(n) 
def subsets(self, nums):
    def build_subsets(start, path):
        res.append(path)
        for i in range(start, len(nums)):
            build_subsets(i+1, path+[nums[i]])

    res = []
    build_subsets(0, [])
    return res
    


def subsets(self, arr):
    res = []
    def helper(arr, i, cur):
        if i == len(arr):
            res.append(cur)
            return
        helper(arr, i+1, cur+[arr[i]])
        helper(arr, i+1, cur)
    helper(arr, 0, [])
    return res

""" iterative solution """

# iterative
# pattern: [[], [1]], 2 => [[],[1], [2], [1,2]]
# T:O(n * 2^n), S: O(n*2^n)
def subsets(self, nums: List[int]) -> List[List[int]]:
    res = [[]]
    #each num concat with res lists to create new list
    for n in nums:
        tmp = []
        for s in res:
            tmp.append(s + [n])
        res += tmp
    return res


""" example: [1,2,3]
[]
[] -> [1]
[] [1] -> [2] [1,2]
[] [1] [2] [1,2] -> [3] [1,3] [2,3] [1,2,3] 
"""


""" bit manipulation solution """


def subsets(self, nums):
    n = len(nums)
    res = []
    for i in range(2**n, 2**(n+1)):
        # 2**(n+1) - 1 can get 11..11
        # generate 00..0 to 11..1
        # bin() function return a string of binary representation starts with string '0b'
        bitmask = bin(i)[3:]
        tmp = []
        for j in range(n):
            if bitmask[j] == '1':
                tmp.append(nums[j])
        res.append(tmp)
    return res


# pythonic way and also faster
def subsets(self, nums):
    n = len(nums)
    res = []
    for i in range(2**n, 2**(n+1)):
        bitmask = bin(i)[3:]
        res.append([nums[j] for j in range(n) if bitmask[j] == '1'])
    return res


# 90
def subsetsWithDup(self, nums):
    def build_comb(start, path):
        res.append(path)
        for i in range(start, len(nums)):
            if i - 1 >= start and nums[i] == nums[i-1]:
                continue
            build_comb(i+1, path+[nums[i]])

    res = []
    nums.sort()
    build_comb(0, [])
    return res


""" iterative solution """
def subsetsWithDup(self, nums: List[int]) -> List[List[int]]:
    res = [[]]
    nums.sort()
    j = 0
    for i in range(len(nums)):
        tmp = []
        # before update the j to previous size, if current is same as prev num, k update to prev's prev size.
        k = j if i-1 >= 0 and nums[i] == nums[i-1] else 0
        # j is res size before current num add to res. j is previous res size.
        j = len(res)
        for s in res[k:]:
            tmp.append(s + [nums[i]])
        res += tmp
        # if j = len(res), then j is size of updated size.
    return res


# 491
# goal: find all increasing sequence. 
# dfs: 
# if path[-1] <= cur, add to res
def findSubsequences(self, nums: List[int]) -> List[List[int]]:
    res = set()
    self.dfs(nums, 0, [], res)
    return list(res)



def dfs(self, nums, cur, path, res):
    if len(path) >= 2:
        res.add(tuple(path))
    
    for i in range(cur, len(nums)):
        if not path or nums[i] >= path[-1]:
            self.dfs(nums, i+1, path + [nums[i]], res)
        



# 216
def combinationSum3(self, k, n):
    def build_comb(start, path, k, n):
        if k == 0 and n == 0:
            res.append(path)
            return

        for i in range(start, 10):
            if n - i < 0:
                break
            build_comb(i+1, path+[i], k-1, n-i)

    res = []
    build_comb(1, [], k, n)
    return res


# 46
# space complexity: n! n factorial.
# using concatenation of array
def permute(self, nums):
    def get_permute(nums, path):
        if len(nums) == 0:
            res.append(path)
            return

        for i in range(len(nums)):
            get_permute(nums[:i]+nums[i+1:], path+[nums[i]])

    res = []
    get_permute(nums, [])
    return res


# check if current element is in path already, skip it if it is.
def permute(self, nums):
    res = []
    self.dfs(nums, [], res)
    return res


def dfs(self,nums, path, res):
    if len(path) == len(nums):
        res.append(path)
        return 
    
    for i in range(len(nums)):
        if nums[i] not in path: 
            self.dfs(nums, path+[nums[i]], res)
        
            

""" iterative solution """
""" 
python slice will not throw error for example a = '1', a[:5] = '1' and a[5:] = '' 
"""

# iterative 肯定是个累积的过程，你有n个数要添加， 把当前数插入每一个插入点 成为一个potential result base， update perms list
def permute(nums):
    perms = [[]]
    # insert each n into result perms
    for n in nums:
        tmp = []
        # for every perm in perms, insert current n into every possible insertion point.
        for p in perms:
            for i in range(len(p)+1):
                tmp.append(p[:i] + [n] + p[i:]) # 插入, p[:i] + [n] + p[i+1:]覆盖 
        # update perms list
        perms = tmp
    return perms


# 47
def permuteUnique(self, nums):
    def get_permute(nums, path):
        if len(nums) == 0:
            res.append(path)
            return
        for i in range(len(nums)):
            if i-1 >= 0 and nums[i] == nums[i-1]:
                continue
            get_permute(nums[:i]+nums[i+1:], path+[nums[i]])
    nums.sort()
    res = []
    get_permute(nums, [])
    return res


# solution not using slicing
# mark the used ones, dont use set to store used items because there are duplicates, store index of used they are unique.
def permuteUnique(self, nums: List[int]) -> List[List[int]]:
        nums.sort()
        res = []
        self.dfs({}, nums, [], res)
        return res
    

def dfs(self, used, nums, path, res):
    if len(path) == len(nums):
        res.append(path)
        return
    
    for i in range(len(nums)):
        if i in used and used[i]: continue 
        if i - 1 >= 0 and nums[i] == nums[i-1] and i-1 in used and not used[i-1]: continue
        used[i] = True
        self.dfs(used, nums, path + [nums[i]], res)
        used[i] = False



""" iterative """
def permuteUnique(self, nums):
    perms = [[]]
    for n in nums:
        tmp = []
        for p in perms:
            # inserting right before i-th position
            for i in range(len(p)+1):
                tmp.append(p[:i] + [n] + p[i:])
                # after insert before i-th position, not on the original list, check if at i-th position is same as the current
                # n, if the same we can terminate because insert after i will be the same as insert before.
                # i < len(p) because there is nothing after i == len(p)
                if i < len(p) and p[i] == n:
                    break
        perms = tmp
    return perms



# 267
# trick: find the middle of the palindrome, and form half of the palindromes. 
def generatePalindromes(self, s: str) -> List[str]:
    res = []
    dic = collections.defaultdict(int)
    char_list, odd = [], 0 
    mid_char = '' 
    # generate char -> cnt
    for c in s: dic[c] += 1
    
    # put half of the chars into char_list
    for k, v in dic.items():
        if v & 1 == 1: 
            mid_char = k
            odd += 1
        char_list += ([k] * (v // 2)) if v & 1 == 0 else [k] * ((v-1)// 2)
    
    if odd > 1: return res
    self.pal_perm(char_list, '', res, len(char_list), mid_char)
    return res 
    

def pal_perm(self, s, path, res, n, mid): 
    if len(path) == n:
        res.append(path + mid + path[::-1])
        return 
    
    for i in range(len(s)):
        if i-1>=0 and s[i] == s[i-1]: continue 
        self.pal_perm(s[:i] + s[i+1:], path + s[i], res, n, mid)
        

# 784
# dfs
def letterCasePermutation(self, S):
    def build_path(i, path):
        if len(path) == len(S):
            res.append(path)
            return
        """ for every position, add swapped case characters to previous path or add original case to previous path or non character to the previous path"""
        if S[i].isalpha():
            # string concatenation will create a new string path thus below concatenation will create a new one
            build_path(i+1, path+S[i].swapcase())
        build_path(i+1, path+S[i])
    res = []
    build_path(0, '')
    return res



""" implement bfs solution """
def letterCasePermutation(self, S):
    queue = collections.deque()
    queue.append(S)
    # every char has two states need to be add the queue, if not char, skip that position
    for i in range(len(S)):
        size = len(queue)
        if S[i].isdigit():
            continue
        while size:
            front = queue.popleft()
            queue.append(front)
            queue.append(front[:i] + front[i].swapcase() + front[i+1:])
            size -= 1
    return list(queue)


""" best and clean and pythonic solution """
def letterCasePermutation(self, S):
    res = ['']
    for c in S:
        if c.isalpha():
            res = [i + j for i in res for j in (c.lower(), c.upper())]
        else:
            res = [i + c for i in res]
    return res


""" python lib """
# itertools.product(alist of iterables) is equal nested loops or cartitian product of two iterable elements
# for example: product([1,2], [3,4])  => [(1,3), (1,4), (2,3), (2,4)]
# *L here meaning more than two iterables need to find cartitian products


def letterCasePermutation(S):
    from itertools import product
    L = [[i.lower(), i.upper()] if i.isalpha() else i for i in S]
    return [''.join(i) for i in product(*L)]


#79
# dfs/backtrack
# board is modified
def exist(self, board, word):
    n, m = len(board), len(board[0])
    for i in range(n):
        for j in range(m):
            if self.dfs(board, i, j, word):
                return True
    return False


def dfs(self, b, i, j, w):
    if not w:
        return True
    if i < 0 or i >= len(b) or j < 0 or j >= len(b[0]) or b[i][j] != w[0]:
        return False

    old = b[i][j]
    b[i][j] = '#'
    for x, y in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
        if self.dfs(b, i+x, j+y, w[1:]):
            return True
    b[i][j] = old
    return False



# better version and not modified
# time: O(n*m*4*L) where L is the length of the word
def exist(self, board: List[List[str]], word: str) -> bool:
    self.d = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    n, m = len(board), len(board[0])
    visited = [[False] * m for _ in range(n)] 
    for i in range(n):
        for j in range(m):
            if self.found(i, j, board, 0, word, visited):
                return True
    return False


def found(self, i, j, board, cur, word, visited):
    if cur == len(word): return True
    if i < 0 or i >= len(board) or j < 0 or j >= len(board[i]) or word[cur] != board[i][j] or visited[i][j]:
        return False
    visited[i][j] = True 
    for x, y in self.d:
        if self.found(x+i, y+j, board, cur+1, word, visited):
            return True
    visited[i][j] = False
    return False
        


# 22
# choices: open and closing parenthesis
# constraints:
# 1. open parenthesis and closing parenthesis must less and equal to n
# 2. valid pair of parenthesis must start with open, thus the open cannot greater than closing parenthesis, thus op <= cl meaning the open paren available must less than or equal to closing paren available
def generateParenthesis(self, n: int) -> List[str]:
        res = []
        left = right = n
        self.backtrack(left, right, [], res)
        return res
    
    
def backtrack(self, left, right, path, res):
    if left < 0: return
    if left == 0 and right == 0: 
        res.append(''.join(path))
        return 
    
    self.backtrack(left-1, right, path +['('], res)
    if left < right:
        self.backtrack(left, right-1, path + [')'], res)
        


""" dp solution """
def generateParenthesis(self, n):
    dp = [[] for i in range(n + 1)]
    dp[0].append('')
    for i in range(n + 1):
        for j in range(i):
            dp[i] += ['(' + x + ')' + y for x in dp[j] for y in dp[i - j - 1]]
    return dp[n]

# other solutions
# dfs/bottom up
# form valid parenthesis pair from bottom up
def generateParenthesis(self, n: int) -> List[str]:
    if n == 0: return []
    if n == 1: return ['()']
    res = set()
    tmp = self.generateParenthesis(n-1)
    for st in tmp:
        for i in range(len(st)):
            res.add(st[:i] + '()' + st[i:])
    return list(res)


# dp
def generateParenthesis(self, n: int) -> List[str]:
    dp = [[] for _ in range(n+1)]
    dp[1].append('()')
    for i in range(2, n+1):
        prev = dp[i-1]
        for st in prev:
            for j in range(len(st)):
                dp[i].append(st[:j] + '()' + st[j:])
        dp[i] = list(set(dp[i]))
    return dp[n]


# iterative
def generateParenthesis(self, n: int) -> List[str]:
    stack = [['()']]
    while n > 1:
        tmp = []
        for st in stack[-1]:
            for j in range(len(st)):
                tmp.append(st[:j] + '()' + st[j:])
        stack.append(list(set(tmp)))
        n -= 1
    return stack[-1]


# 93
def restoreIpAddresses(self, s):
    def build_ip(s, k, path):
        if k == 0:
            # when k is 0, there could be some digits left behind, so if nothing left we have a valid ip
            if not s:
                path = path[:-1]
                res.append(path)
                return

        # each stack frame we have 3 choices, choose 1, 2, 3 digits
        for i in range(1, 4):
            # the substring s should always at least i long, this condition easy to forget.
            if i <= len(s):
                if i == 1:
                    build_ip(s[i:], k-1, path+s[:i]+'.')
                # if choose 2 or 3 digits, the string cannot start with 0
                elif i == 2 and s[0] != '0':
                    build_ip(s[i:], k-1, path+s[:i]+'.')
                elif i == 3 and s[0] != '0' and int(s[:i]) <= 255:
                    build_ip(s[i:], k-1, path+s[:i]+'.')

    res = []
    if len(s) > 12:
        return res
    build_ip(s, 4, '')
    return res

# without modify s version


def restoreIpAddresses(self, s: str) -> List[str]:
    res = []
    self.build(0, s, '', 4, res)
    return res


def build(self, start, s, path, n, res):
    if start > len(s) or n < 0:
        return
    if start == len(s):
        if n == 0:
            res.append(path[:-1])
        return

    for i in range(1, 4):
        cur = s[start:start+i]
        if i == 1:
            self.build(start + 1, s, path + cur + '.', n-1, res)
        elif i == 2 and cur[0] != '0':
            self.build(start + 2, s, path + cur + '.', n-1, res)
        elif i == 3 and cur[0] != '0' and int(cur) <= 255:
            self.build(start + 3, s, path + cur + '.', n-1, res)




# 131
# backtrack
# choices: 0<= i < len(s)
# terminate: if not s
# check s[:i] is pal and partition(s[i:])
# T(n)=T(n-1)+T(n-2)+..+T(1)
# T(n+1)=T(n)+T(n-1)+..+T(1)
# T(n+1)=2T(n)  -> input size increase by 1, the total time doubled.
# T(n)=2^n
# T: O(2^n), S:(2^n)
def partition(self, s):
    res = []
    self.dfs(s, [], res)
    return res


def dfs(self, s, path, res):
    if not s and path:
        res.append(path)
        return

    for i in range(1, len(s)+1):
        if s[:i] == s[:i][::-1]:
            self.dfs(s[i:], path + [s[:i]], res)


# solution that maintain only one temp array and the end copy all the pal and add to result
def partition(self, s):
    def build_comb(s, start, path):
        if start == len(s):
            tmp = path[:]
            res.append(tmp)
            return

        for i in range(start, len(s)):
            if s[start:i+1] == s[start:i+1][::-1]:
                path.append(s[start:i+1])
                build_comb(s, i+1, path)
                path.pop()

    res = []
    if not s:
        return res
    build_comb(s, 0, [])
    return res




# 698
""" two ways to consider what choices are for each stack frame
1. for each element, there are k buckets to put, then the goal is to use out all the elements to fill all the buckets
2. for each bucket, there is a list of items to select, then the goal is fill each buckets with items in the list
"""
# choices: k same fix-sized buckets for every items in nums, to put element one by one
# constraints: each bucket must have enough space to fill the number
# goal: place each number into k buckets


def canPartitionKSubsets_sorted(self, nums, k):
    if len(nums) < k:
        return False
    s = sum(nums)
    # naturally you put the bigger object into the buckets first, this will give you more flexibility.
    nums.sort(reverse=True)
    if s % k != 0:
        return False
    # create k equal size buckets
    sub = s//k
    buckets = [sub]*k

    # pos is current working element position
    def fill_buckets(pos):
        if pos == len(nums):
            return True
        # go thru each bucket find the bucket that can fill the element at pos and successfully fill the rest of buckets
        # otherwise, you cannot put pos element at current bucket so you need to take it out.
        for i in range(k):
            if buckets[i] >= nums[pos]:
                buckets[i] -= nums[pos]
                if fill_buckets(pos+1):
                    return True
                buckets[i] += nums[pos]
                # if buckets[i] == sub means that its impossible to fill buckets[i] so no need to try rest of buckets.
                if buckets[i] == sub:
                     break
        return False
    return fill_buckets(0)


# choices: for each bucket, choice is nums , to fill bucket one by one,
# constraints: each bucket must have enough space to fill the number
# goal: fill each bucket
def canPartitionKSubsets(self, nums, k):
    if len(nums) < k:
        return False
    s = sum(nums)
    if s % k != 0:
        return False
    target = s//k
    # to mark visted elements
    visited = [False for _ in range(len(nums))]

    # num_cnt: denotes number of elemnets inside current bucket
    def dfs(num_cnt, start, k, total):
        if k == 1:
            return True
        # if total meets the target and number of elemnts inside current buckets more than 0, we can move on to rest k-1 buckets
        # reset start index back to 0 coz some of the elements may not be visited previously
        if total == target and num_cnt > 0:
            return dfs(0, 0, k-1, 0)
        # for current stack, we have choices from [start, len(nums)]
        for i in range(start, len(nums)):
            if not visited[i]:
                visited[i] = True
                # if current element is not visted, add to the total and increment the start index and num_cnt
                if dfs(num_cnt+1, i+1, k, total+nums[i]):
                    return True
                visited[i] = False
        return False

    return dfs(0, 0, k, 0)




# 127
# BFS
# goal: min steps transform to endword
# nei: 25 letters (besides self)
# constraints: each transform must in word list
# terminate: cur_word == endword
# avoid repeating, remove transformed word from wordlist, or use visited set
# T:O(26 * L * L * M), S: O(M*M)
def ladderLength(self, beginWord, endWord, wordList):
    dic = set(wordList)
    queue = collections.deque()
    queue.append([beginWord, 1])
    while queue:
        w, length = queue.popleft()
        for i in range(len(w)):
            for c in string.ascii_lowercase:
                tmp = w[:i] + c + w[i+1:]
                if tmp not in dic:
                    continue
                if tmp == endWord:
                    return length+1
                queue.append([tmp, length+1])
                dic.remove(tmp)
    return 0


""" two-end bfs """
def ladderLength(self, beginWord, endWord, wordList):
    words = set(wordList)
    front, back = {beginWord}, {endWord}
    level = 1
    while front:
        tmp = set()
        level += 1
        for w in front:
            for i in range(len(w)):
                for c in string.ascii_lowercase:
                    if c != w[i]:
                        cur = w[:i] + c + w[i+1:]
                        # meaning two end bfs meet at same word, then current level is the result
                        if cur in back:
                            return level
                        if cur in words and cur != endWord:
                            tmp.add(cur)
                            words.remove(cur)
        front = tmp
        # always pick the smaller group to process
        if len(front) > len(back):
            front, back = back, front
    return 0


# 241
# divide and conquar
# key is recursively calcuate the cartitian products of two partitions
def diffWaysToCompute(self, input):
        if input.isdigit():
            return [int(input)]

        res = []  # be careful not to put this inside the loop
        # iterate thru the input, and partition the input into two parts for every operator.
        for i in range(len(input)):
            if input[i] in '+-*':
                # divide and conquar
                res1 = self.diffWaysToCompute(input[:i])
                res2 = self.diffWaysToCompute(input[i+1:])

                for n in res1:
                    for m in res2:
                        res.append(self.process(n, m, input[i]))
        return res


def process(self, left, right, op):
        if op == '+':
            return left + right
        elif op == '-':
            return left - right
        else:
            return left * right


# 842
def splitIntoFibonacci(self, S):
        n = len(S)
        # try all the length of first two numbers.
        # according the description, each number is at most 10 digits long, size of the integer represetation
        for i in range(1, 11):
            for j in range(1, 11):
                # if first two num length is equal or longer than the length of the input, then not able to form fib, so we try another pairs
                if i + j >= n:
                    continue
                res = self.build_fib(i, j, S)
                # res could be empty list
                if len(res) >= 3:
                    return res
        return []


def build_fib(self, i, j, s):
    a, b = s[:i], s[i:i+j]
    # check if a number is start with 0
    if a[0] == '0' and i > 1:
        return []
    if b[0] == '0' and j > 1:
        return []

    n = len(s)
    first, second = int(a), int(b)
    arr = [first, second]
    offset = i + j
    while offset < n:
        tmp = first + second
        third = str(tmp)
        k = len(third)

        if third != s[offset: offset+k]:
            return []
        third = int(third)
        if third > pow(2, 31)-1:
            return []
        arr.append(third)
        offset += k
        first, second = second, third
    return arr


""" backtracking solution """


def splitIntoFibonacci_dfs(self, S):
    res = []
    self.can_split(S, 0, res)
    return res


def can_split(self, s, start, path):
    if start == len(s) and len(path) >= 3:
        return True

    for i in range(start, len(s)):
        if i > start and s[start] == '0':
            break

        num = int(s[start: i+1])

        if num > 2 ** 31 - 1:
            break

        size = len(path)
        if size >= 2 and num > path[size-1] + path[size-2]:
            break
        # here we cannot code it like this ' path + [num] ' becaue this will create a new list object assgin to the paremeter res.
        # thus this res is not pointing to the same object as the calling function res.
        if size < 2 or num == path[size-1] + path[size-2]:
            path.append(num)
            if self.can_split(s, i+1, path):
                return True
            path.pop()

    return False


# 301
# TLE
class Solution:
    def removeInvalidParentheses(self, s):
        if not s:
            return ['']
        l, r = self.invalid_count(s)
        res = []
        self.dfs(s, l, r, res)
        return res

    # """ smart way to count unbalanced open and closing paren """
    def invalid_count(self, s):
        l, r = 0, 0
        for c in s:
            l += (c == '(')
            """ when opening paren is 0 and current paren is closing, then the unbalanced closing should increment by 1 
            if opening unbalanced paren is not 0, and current paren is closing paren, then meaning they are matched, so decrese openning 
            unbalanced paren count by 1 
            """
            if l == 0:
                r += (c == ')')
            else:
                l -= (c == ')')
        return [l, r]

    """ simple way to check if paren are matched for not, we do not need to use stack to do it when there is only one type of paren  """

    def is_valid(self, s):
        count = 0
        for c in s:
            if c == '(':
                count += 1
            elif c == ')':
                count -= 1
            # whenever count is less than 0 during the loop, meaning there is unbalanced closing paren
            if count < 0:
                return False
        return count == 0

    def dfs(self, s, l, r, res):
        if l == 0 and r == 0:
            if self.is_valid(s) and s not in res:
                res.append(s)
            return

        for i in range(len(s)):
            if i > 0 and s[i] == s[i-1]:
                continue
            if s[i] in '()':
                if r > 0:
                    self.dfs(s[:i]+s[i+1:], l, r-1, res)
                elif l > 0:
                    self.dfs(s[:i]+s[i+1:], l-1, r, res)


# 212
# TLE
class Solution:
    def findWords(self, board, words):
        res = []
        for w in words:
            if self.exists(board, w):
                res.append(w)
        return res

    def exists(self, b, w):
        if not b:
            return False
        n, m = len(b), len(b[0])
        for i in range(n):
            for j in range(m):
                if self.find_path(i, j, b, w, 0):
                    return True
        return False

    def find_path(self, i, j, b, w, k):
        if i < 0 or i >= len(b) or j < 0 or j >= len(b[0]) or b[i][j] != w[k]:
            return False

        if k == (len(w)-1):
            return True

        tmp = b[i][j]
        b[i][j] = '#'
        found = (self.find_path(i+1, j, b, w, k+1) or
                 self.find_path(i-1, j, b, w, k + 1) or
                 self.find_path(i, j+1, b, w, k+1) or
                 self.find_path(i, j-1, b, w, k+1))
        """ for this problem, we need to reuse the same board, so for the true case, we also need to recover the # back to original value """
        b[i][j] = tmp
        return found


""" implement trie solution """



#36 
def isValidSudoku(self, board: List[List[str]]) -> bool:
    for i in range(9):
        row , col, cube = set(), set(), set()
        for j in range(9):
            if board[i][j] != '.':
                if board[i][j] in row:
                    return False
                row.add(board[i][j])
                
            if board[j][i] != '.':
                if board[j][i] in col:
                    return False
                col.add(board[j][i])
                
            start_row = 3 * (i // 3)
            start_col = 3 * (i % 3)
            cur = board[start_row + j//3][start_col + j%3]
            if cur != '.': 
                if cur in cube:
                    return False
                cube.add(cur)
    return True



# 37
class Solution:
    def solveSudoku(self, board):
        self.fill(board)

    def fill(self, board):
        for i in range(9):
            for j in range(9):
                if board[i][j] == '.':
                    for k in ['1', '2', '3', '4', '5', '6', '7', '8', '9']:
                        # if k is valid for current cell, then fill the rest, if it works return true else, put '.' back and try other digits
                        if self.is_valid(board, i, j, k):
                            board[i][j] = k
                            if self.fill(board):
                                return True
                            else:
                                board[i][j] = '.'
                    return False
        return True

    def is_valid(self, board, i, j, k):
        for x in range(9):
            """ there are some . in the cells also.
            the condition not false including cur cell is . or cur cell not equal to k 
            so the opposite is board[i][x] != '.' and board[i][x] == k
            """
            #check col
            if board[i][x] != '.' and board[i][x] == k:
                return False
            #check row
            if board[x][j] != '.' and board[x][j] == k:
                return False

            #check box
            """ think about the subbox as flat-out array
            3 * (i // 3) to get the starting row, 3 * (j // 3) to get starting colomn 
            x//3 to increment the row, when x <= 3 row will not increment,
            x%3 to find the colmn 
            """
            if board[3 * (i // 3) + x // 3][3 * (j // 3) + x % 3] != '.' and board[3 * (i // 3) + x // 3][3 * (j // 3) + x % 3] == k:
                return False
        return True

# clean version


class Solution(object):
    def solveSudoku(self, board):
        self.fill(board)

    def fill(self, board):
        for i in range(9):
            for j in range(9):
                if board[i][j] == '.':
                    for d in ['1', '2', '3', '4', '5', '6', '7', '8', '9']:
                        if self.check(i, j, board, d):
                            board[i][j] = d
                            if self.fill(board):
                                return True
                            board[i][j] = '.'
                    return False
        return True

    def check(self, i, j, board, d):
        for x in range(9):
            if board[x][j] == d:
                return False
            if board[i][x] == d:
                return False
            if board[3 * (i // 3) + x // 3][3 * (j // 3) + x % 3] == d:
                return False
        return True


""" bitmask solution optimized """
# https: // leetcode.com/problems/sudoku-solver/discuss/15796/Singapore-prime-minister-Lee-Hsien-Loong's-Sudoku-Solver-code-runs-in-1ms


# 51
class Solution(object):
    def solveNQueens(self, n):
        res = []
        queen_cols = [-1]*n  # marks the col i-th queen is placed
        self.dfs(n, 0, queen_cols, [], res)
        return res

    # idx means row and also means idx-th queen
    # using call stack frames to represent rows
    def dfs(self, n, idx, cols, path, res):
        # goal is placing n queens in n x n board
        if idx == n:
            res.append(path)
            return
        # choices are n columns, each queen will take a column
        for i in range(n):
            # means put idx-th queen at column i
            cols[idx] = i
            if self.is_valid(cols, idx):
                tmp = '.' * n
                self.dfs(n, idx+1, cols, path +
                         [tmp[:i] + 'Q' + tmp[i+1:]], res)

    # check vertical and diagonal
    def is_valid(self, cols, r):
        for i in range(r):
            ''' 
            cols[i] == cols[r] means r-th queen col is the same as i-th queen col
            abs(cols[i] - cols[r]) == r - i means r-th queen and i-th queen col and row distance are the same, thus they are on the same diagnals.
            '''
            if cols[i] == cols[r] or abs(cols[i] - cols[r]) == r - i:
                return False
        return True


# 282
class Solution(object):
    def addOperators(self, num, target):
        """
        :type num: str
        :type target: int
        :rtype: List[str]

         goal: make path result to target
         choices: s[:i] where i in [1, n+1]
  
         prev: last multiply value
         val: current val
         path: res string
        """
        res = []
        self.build_path(num, '', 0, None, res, target)
        return res

    def build_path(self, s, path, val, prev, res, trgt):
        if not s and val == trgt:
            res.append(path)
            return

        for i in range(1, len(s)+1):
            tmp = int(s[:i])
            # prevent starting '01'
            if i == 1 or (i > 1 and s[0] != '0'):
                # cannot write 'if not prev' because prev == 0 will also make this condition work, this way will fail case '105'
                # it will return 1*05, the val is updated correctly, but coz the above condition, will execute if block instead it should
                # execute else block, thus causing miss operator.
                if prev is None:
                    # add first number into path
                    self.build_path(s[i:], path + s[:i],val + tmp, tmp, res, trgt)
                else:
                    self.build_path(s[i:], path + '+' + s[:i],val + tmp, tmp, res, trgt) 
                                    
                    # need to update the prev with sign, -tmp
                    self.build_path(s[i:], path + '-' + s[:i],val - tmp, -tmp, res, trgt)
                                    
                    # update the prev, and subtract prev from val, then calculate new val
                    self.build_path(s[i:], path + '*' + s[:i],val - prev + prev*tmp, prev*tmp, res, trgt)
                                    



# 934
""" # idea hear is that find the fist connected component using dfs and then using bfs to expand this component, once meet the other 
# component return total steps took to get there.  """


class Solution:
    def __init__(self):
        self.directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        import collections
        self.queue = collections.deque()

    def shortestBridge(self, A):
        n, m = len(A), len(A[0])
        visited = [[False for _ in range(m)] for _ in range(n)]
        self._fill(A, visited)
        steps = 0
        """ bfs """
        while self.queue:
            # level size
            size = len(self.queue)
            while size:
                i, j = self.queue.popleft()
                size -= 1
                for x, y in self.directions:
                    # be careful here dont write it like i += x or j += y, this way will completely changed the i an j's value
                    # coz we need to to visit all neibhours of node[i][j]
                    x += i
                    y += j
                    if 0 <= x < len(A) and 0 <= y < len(A[0]) and not visited[x][y]:
                        if A[x][y] == 0:
                            visited[x][y] = True
                            self.queue.append((x, y))
                        else:
                            return steps
            # increment by 1 when all nodes in the same level is traversed.
            steps += 1
        return -1

    def _fill(self, b, visited):
        n, m = len(b), len(b[0])
        for i in range(n):
            for j in range(m):
                if b[i][j] == 1:
                    # try to find the first component and then return
                    self._dfs(i, j, b, visited)
                    return

    def _dfs(self, i, j, b, visited):
        if i < 0 or i >= len(b) or j < 0 or j >= len(b[0]) or visited[i][j] or b[i][j] != 1:
            return
        """ this line add all the nodes in the compoent to the queue
        here is the trick, it treats all nodes in the component as one level or as neibhours of each other """
        self.queue.append((i, j))
        visited[i][j] = True
        for x, y in self.directions:
            self._dfs(i+x, j+y, b, visited)



# 752
# basic version bfs 
def openLock(self, deadends, target):
        # there may be more than lock states that can produce same lock states, thus you need need mark which ones are visited before.
        deads, visited = set(deadends), set()
        visited.add('0000')
        import collections
        queue = collections.deque()
        queue.append('0000')
        level = 0
        while queue:
            size = len(queue)
            while size:
                """ process block: proces root node here """
                front = queue.popleft()
                # means deadends, you cannot use this to turn forward or backward
                if front in deads:
                    size -= 1
                    continue

                if front == target:
                    return level
                """ end of process block """
                # one move on one slot include up and down. there are 4 slots either roll up or roll down
                """ adding neigbors block """
                for i in range(4):
                    roll_down = front[:i] + ('0' if front[i] == '9' else str(int(front[i])+1)) + front[i+1:]
                    roll_up = front[:i] + ('9' if front[i] == '0' else str(int(front[i])-1)) + front[i+1:]

                    if roll_down not in visited:
                        visited.add(roll_down)
                        queue.append(roll_down)

                    if roll_up not in visited:
                        visited.add(roll_up)
                        queue.append(roll_up)
                size -= 1
                """ end of adding neigbors block """
            level += 1
        return -1



# cleaner version of one way bfs using set because the order of processing does not matter.
def openLock(self, deadends, target):
    queue, deads= set(['0000']), set(deadends)
    turn = 0
    while queue:
        tmp = set()
        for front in queue:
            if front in deads: continue
            if front == target: return turn
            deads.add(front)
            for i in range(len(target)):
                fw = str((int(front[i])+1) if front[i] != '9' else 0)
                bw = str((int(front[i])-1) if front[i] != '0' else 9)
                forward = front[:i] + fw + front[i+1:]
                backward = front[:i] + bw + front[i+1:]     
                if forward not in deads: tmp.add(forward)
                if backward not in deads: tmp.add(backward)
        queue = tmp
        turn += 1
    return -1
                

# fast two-end bfs
# in this case the order of processing the lock state does not matter, thus we can use set to contain all the states
def openLock(self, deadends, target):
    deads = set(deadends) # to hold deads and visited lock states
    begin, end = set(['0000']), set([target])
    turn = 0
    while begin and end:
        # always pick the smaller sets to search and add states 
        if len(begin) > len(end): 
            begin, end = end, begin
        tmp = set()
        for s in begin:
            # if s in end meaning one side already visited this lock state, thus they meet at this state
            if s in end: return turn
            if s in deads: continue
            deads.add(s)
            for i in range(len(target)):
                fw = str((int(s[i])+1) if s[i] != '9' else 0)
                bw = str((int(s[i])-1) if s[i] != '0' else 9)
                forward = s[:i] + fw + s[i+1:]
                backward = s[:i] + bw + s[i+1:]  
                if forward not in deads: tmp.add(forward)
                if backward not in deads:  tmp.add(backward)
        turn += 1          
        begin = tmp
    return -1
                    



# 133
class Node:
    def __init__(self, val=0, neighbors=[]):
        self.val = val
        self.neighbors = neighbors


class Solution:
    """ dfs """
    def cloneGraph(self, node):
        if not node:
            return
        node_cp = Node(node.val)
        cloned = {node: node_cp}
        self.dfs(node, cloned)
        return node_cp

    # no need to use visited set we can simply use the dic 
    def dfs(self, node, cloned):
        for nei in node.neighbors:
            if nei not in cloned:
                nei_cp = Node(nei.val)
                cloned[nei] = nei_cp
                cloned[node].neighbors.append(nei_cp)
                self.dfs(nei, cloned)
            else:
                #the nei node may be visted before but the relation between nei and its parent has not been established in the cloned version graph, 
                # or the edge between the current node and this visited neighbor in the cloned graph has not been established yet
                cloned[node].neighbors.append(cloned[nei])




class Solution(object):
    """ bfs """
    def cloneGraph(self, node):
        if not node:return
        node_cp = Node(node.val)
        cloned = {node: node_cp}
        self.bfs(node, cloned)
        return node_cp


    def bfs(self, node, cloned):
        import collections
        queue = collections.deque()
        queue.append(node)
        while queue:
            n = queue.popleft()
            for nei in n.neighbors:
                if nei not in cloned:
                    nei_cp = Node(nei.val)
                    cloned[nei] = nei_cp
                    cloned[n].neighbors.append(nei_cp)
                    queue.append(nei)
                else:
                    cloned[n].neighbors.append(cloned[nei])


# 547
""" connected component in undirected graph. use dfs to find all related friends of current student
tip: for N*N matrix, there are N students. so we need to find their relations. 
because it's a undirected graph, we need to avoid infinit loop by mark the visited student. for example student 
A is frined of B, B is friend of A. if dont mark A as visited, A will be processed again from B. 
"""

class Solution(object):
    def findCircleNum(self, M):
        student_count = len(M)
        cycle_count = 0
        visited = [False] * student_count

        for student in range(student_count):
            if not visited[student]:
                visited[student] = True
                self.find_friends(M, student, visited)
                cycle_count += 1
        return cycle_count

    def find_friends(self, M, student, visited):
        for another_student in range(len(M)):
            if (M[student][another_student] == 1) and (not visited[another_student]):
                visited[another_student] = True
                self.find_friends(M, another_student, visited)


# union find solution
# output number of connected components
def findCircleNum(self, M: List[List[int]]) -> int:
    n  = len(M)
    ID = [i for i in range(n)]
    size = [1] * n
    comp_cnt = n
    for i in range(n):
        for j in range(n): 
            if i != j and M[i][j]  == 1:
                if self.find(i, ID) != self.find(j, ID):
                    self.union(i, j, ID, size)
                    comp_cnt -= 1 
    return comp_cnt
    
    

def union(self, p, q, ID, size):
    p, q = self.find(p, ID), self.find(q, ID)
    if size[p] < size[q]:
        ID[p] = q
        size[q] += size[p]
    else:
        ID[q] = p
        size[p] += size[q]


def find(self, p, ID):
    while p != ID[p]:
        ID[p] = ID[ID[p]]
        p = ID[p]
    return p
        
        




# 695
def maxAreaOfIsland(self, grid):
    n, m = len(grid), len(grid[0])
    max_area = 0
    for i in range(n):
        for j in range(m):
            max_area = max(max_area, self.dfs(grid, i, j))
    return max_area


def dfs(self, g, i, j):
    # 0 will have 0 area return
    if i < 0 or i >= len(g) or j < 0 or j >= len(g[0]) or g[i][j] != 1:
        return 0

    g[i][j], cnt = 0, 0

    for x, y in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        cnt += self.dfs(g, i+x, j+y) # do not add 1 here. this is the total of four directions's 1s. 
    return cnt + 1 # plus current node



# 733
def floodFill(self, image, sr, sc, newColor):
        if image[sr][sc] == newColor:
            return image
        self.dfs(image, sr, sc, image[sr][sc], newColor)
        return image


def dfs(self, img, i, j, old, new):
    if i < 0 or i >= len(img) or j < 0 or j >= len(img[0]) or img[i][j] != old:
        return

    img[i][j] = new
    for x, y in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        self.dfs(img, i+x, j+y, old, new)


# 841
# goal: check if can enter every room
# dfs: visited all rooms reachable
# tip: backtrack is used for finding all paths, dfs can be used to visited every possible node in the graph.
# this problem ask if every room can be reached, thus need to use dfs. 
# if the problem ask if starting from 0, if there is a path to visit every room, then use backtrack.
# T: O(rooms+Keys), S: O(rooms)
def canVisitAllRooms(self, rooms: List[List[int]]) -> bool:
    visited = set()
    self.dfs(0, rooms, visited)
    return len(visited) == len(rooms)
    
# kind like preorder traversal   
def dfs(self, start, rooms, visited):
    visited.add(start)
    for k in rooms[start]:
        if k not in visited: 
            self.dfs(k, rooms, visited)



# 1202
""" tip: find all the components chars and their indices, assign the small char to the small index """
# also could be soved using union find


class Solution(object):
    def smallestStringWithSwaps(self, s, pairs):
        n = len(s)
        adj = [[] for _ in range(n)]
        # build the ajacency list/bags
        for i, j in pairs:
            adj[i].append(j)
            adj[j].append(i)

        visited = [False] * n
        s_list = list(s)

        for i in range(n):
            component = []
            self.dfs(i, adj, visited, component)
            component.sort()  # sort indices later, the small char will take the index in order
            chars = [s_list[j] for j in component]
            chars.sort()
            for k in range(len(component)):
                s_list[component[k]] = chars[k]
        return ''.join(s_list)

    def dfs(self, i, adj, visited, comp):
        visited[i] = True
        comp.append(i)
        for j in adj[i]:
            if not visited[j]:
                self.dfs(j, adj, visited, comp)


""" union find implementation """


# 1162
""" bfs, find all the 1s and expanding outward simultanously """
class Solution(object):
    def maxDistance(self, grid):
        n = len(grid)
        import collections
        queue = collections.deque()
        # add all the lands to the queue as if they are neighbors.
        for i in range(n):
            for j in range(n):
                if grid[i][j] == 1:
                    queue.append((i, j))  # all 1s to the queue

        if len(queue) == n*n or len(queue) == 0:
                return -1

        level = 0
        while queue:
            size = len(queue)
            while size:
                i, j = queue.popleft()
                size -= 1
                for x, y in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    x += i
                    y += j
                    # expand the land level by level until not water reached
                    if 0 <= x < n and 0 <= y < n and grid[x][y] == 0:
                        grid[x][y] = 1
                        queue.append((x, y))
            level += 1
        return level-1


# 802
# TLE: because algorithm only stores those unsafe positions after calling hasCycle function but does not safe the positions that is safe
def eventualSafeNodes(self, graph: List[List[int]]) -> List[int]:
    res = []
    N = len(graph)
    seen = set()
    for n in range(N):
        if n not in seen and not self.hasCycle(graph, n, seen):
            res.append(n)
    return res

    
    
def hasCycle(self, graph, cur, seen):
    if cur in seen:
        return True
    seen.add(cur)
    for nei in graph[cur]:
        if self.hasCycle(graph, nei, seen):
            return True
    seen.remove(cur)
    return False


        
# a node is eventually safe means that all its paths should arrive at terminatal nodes.
# a terminal is node with no out degrees.
# if node i has path that creates cycle, node i is not eventual safe.
def eventualSafeNodes(self, graph: List[List[int]]) -> List[int]:
    res  = []
    visited = [None] * len(graph)
    for i in range(len(graph)):
        if not self.hasCycle(graph, i,visited):
            res.append(i)
    return res

# algorithm neeeds to handle three states, safe, unsafe, unvisited
# use None to represent unvisited
    
# detect cycle, return boolean, collect each visited node state.
def hasCycle(self, graph, cur, visited):
    if visited[cur] != None: 
        return visited[cur] == "unsafe"
    
    visited[cur] = "unsafe"
    for to in graph[cur]:
        if self.hasCycle(graph, to, visited):
            return True
    visited[cur] = "safe"
    return False
        


# 207
# cycle detection 
def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
    dic = collections.defaultdict(list)
    for c, pre in prerequisites:
        dic[pre].append(c)
    states = [None] * numCourses
    # check if i node's subgraph has cycle, if cycle exits, cannot finish the courses.
    for i in range(numCourses):
        if states[i] == 'yes' or self.hasCycle(i, states, dic):
            return False
    return True

# backtracking
def hasCycle(self, i, states, dic):
    if states[i]: 
        return states[i] == 'yes'
    states[i] = 'yes'
    for j in dic[i]:
        if self.hasCycle(j, states, dic):
            return True
    states[i] = 'no'
    return False


# topological sort. bfs level 是根据先后顺序来的 , topo 的上下关系很重要
# time O(E + V): use O(E) to create relation graph, where E is the size of prerequisite, O(v) to traverse all courses. 
def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
    # pre -> [next]
    out_degree = [0] * numCourses
    # cur -> [prev]
    graph = [[] for _ in range(numCourses)]
    
    for cur, pre in prerequisites:
        out_degree[pre] += 1 
        graph[cur].append(pre)

    # find all courses that are not prerequisites for any other courses
    queue = collections.deque([i for i in range(numCourses) if out_degree[i] == 0])
            
    # count: number of course visited
    count = len(queue)
    # bfs
    while queue:
        cur = queue.popleft()
        for pre in graph[cur]:
            out_degree[pre] -= 1 
            # meaning all the upper level nodes are handled, 如果有cycle， queue 就会提前停止， 因为out_degree[pre] 无法为0
            if out_degree[pre] == 0:
                queue.append(pre)
                count += 1 
    return count == numCourses
                

# 1192
# tarjan + cycle detection 
def criticalConnections(self, n: int, connections: List[List[int]]) -> List[List[int]]:
    graph = [[] for _ in range(n)]
    for x, y in connections: 
        graph[x].append(y)
        graph[y].append(x)

    ranks = [-2] * n
    connections = set(map(tuple, map(sorted, connections)))
    self.dfs(0, 0, n, ranks, graph, connections)
    print(connections)
    return list(connections)
    

def dfs(self, cur, d, n, ranks, graph, connections):
    if ranks[cur] >= 0: 
        return ranks[cur]
    ranks[cur] = d
    min_rank = n
    for nei in graph[cur]: 
        if ranks[nei] == d-1: 
            continue
        rank = self.dfs(nei, d+1, n, ranks, graph, connections)
        if rank <= d:
            # discard will not raise an error when the item is not exit in the set
            connections.discard(tuple(sorted([cur, nei])))
        min_rank = min(min_rank, rank)
    return min_rank




# 210
# dfs build path bottom up
def findOrder(self, numCourses, prerequisites):
    indegrees = [[] for _ in range(numCourses)]
    for x, y in prerequisites:
        indegrees[x].append(y)

    states = ('can', 'cannot')
    mode = [None] * numCourses
    res = []
    # starting choices are 0 - numCourses-1
    for i in range(numCourses):
        # if cycle detected, return []
        if not self.can_finish(indegrees, mode, i, res, states): return []
    return res


def can_finish(self, g, mode, i, res, states):
    if mode[i]: return mode[i] == states[0]

    mode[i] = states[1]
    for j in g[i]:
        if not self.can_finish(g, mode, j, res, states):
            return False
    mode[i] = states[0]
    res.append(i)
    return True


""" implement BFS  """
def findOrder(self, numCourses: int, prerequisites: List[List[int]]) -> List[int]:
    out_degree = [0] * numCourses
    graph = [[] for _ in range(numCourses)]
    for cur, pre in prerequisites:
        out_degree[pre] += 1
        graph[cur].append(pre)
    #starts from courses that are not prerequisites for any other courses
    queue = collections.deque([i for i in range(numCourses) if out_degree[i] == 0])
    res = []
    count = len(queue)
    while queue:
        course = queue.popleft()
        res.append(course)
        for pre in graph[course]:
            out_degree[pre] -= 1
            if out_degree[pre] == 0:
                count += 1 
                queue.append(pre)
    return res[::-1] if count == numCourses else []


# 269 
# using indegree instead of outdegree is more intuitive
def alienOrder(self, words: List[str]) -> str:
    # left -> right
    graph = collections.defaultdict(set)
    # cur -> cnt
    in_degree = {}
    chars = set()
    for w in words:
        chars |= set(w)

    for c in chars: 
        in_degree[c] = 0
    
    for w1, w2 in zip(words, words[1:]):
        if len(w2) < len(w1) and w1.startswith(w2): return ''# handle wrong leetcode testcase 
        self.compareOrder(w1, w2, graph, in_degree)
    return self.getOrder(graph, in_degree)
    
            
def compareOrder(self, w1, w2, graph, in_degree):
    cur = 0
    while cur < len(w1):
        if w1[cur] != w2[cur]:
            if w2[cur] not in graph[w1[cur]]:
                graph[w1[cur]].add(w2[cur])
                # only increment right node's indegree count when left node does not have right node in its set.
                in_degree[w2[cur]] += 1
            break
        cur += 1
    
        
def getOrder(self, graph, in_degree):
    output = []
    queue = collections.deque([c for c in in_degree if in_degree[c] == 0])
    while queue:
        c = queue.popleft()
        output.append(c)
        for d in graph[c]:
            in_degree[d] -= 1
            if in_degree[d] == 0:
                queue.append(d)
    return ''.join(output) if len(output) == len(in_degree) else ''

        



# 138
class Node:
    def __init__(self, x, next=None, random=None):
        self.val = int(x)
        self.next = next
        self.random = random


class Solution(object):
    # tip: hand next first then handle random using hashtable\
    def copyRandomList(self, head):
        if not head:
            return head
        dic = {}
        n = m = head
        # copy all nodes
        while n:
            dic[n] = Node(n.val)
            n = n.next
        # linked them up
        while m:
            """ tip: 
            get() will not throw key error like [] when accessing dictionary
            for this problem, last node will have None as next, thus if dic[None] will throw key error 
            instead get() will return None if None key is given"""
            dic[m].next = dic.get(m.next)
            dic[m].random = dic.get(m.random)
            m = m.next
        return dic[head]




'''tree problems below'''
# 100
def isSameTree(self, p: TreeNode, q: TreeNode) -> bool:
    stack = [(p, q)]
    while stack:
        p, q = stack.pop()
        if not p and not q: continue
        elif not p or not q: return False
        else: 
            if p.val != q.val: 
                return False
            else:
                stack.append((p.left, q.left))
                stack.append((p.right, q.right))
    return True

                 
# 104 
# iterative way to find depth. 
# trick: store tuple of node and depth 
def maxDepth(self, root: TreeNode) -> int:
    stack = []
    ans = depth = 0 
    while stack or root:
        while root:
            depth+=1
            stack.append((depth, root))
            root = root.left

        if stack:
            depth, root = stack.pop()
            ans = max(ans, depth)
            root = root.right
    return ans 



# 894
# tip: either one node or three nodes to form full binary tree
def allPossibleFBT(self, N: int) -> List[TreeNode]:
    if N == 1: return [TreeNode(0)]

    res = []
    # reserve for root
    N -= 1 
    # increment 2 each time because you need 1, 3, 5.. to form full binary tree.
    for i in range(1, N, 2):
        left, right = self.allPossibleFBT(i), self.allPossibleFBT(N-i)
        # cartitian product of two lists of full binary trees.
        for lt in left:
            for rt in right:
                cur = TreeNode(0)
                cur.left, cur.right = lt, rt
                res.append(cur)
    return res



# 662
# trick: using same idea as the heap. the position relation between parent and children 
def widthOfBinaryTree(self, root: TreeNode) -> int:
    max_width = 0
    queue = collections.deque([(root,1)])
    while queue:
        max_width = max(queue[-1][1] - queue[0][1]+1,  max_width)
        size = len(queue)
        for _ in range(size):
            r, idx = queue.popleft()
            
            if r.left: queue.append((r.left, 2*idx))
            if r.right: queue.append((r.right, 2*idx+1))
    return max_width



# 865
# if a node's two subtrees has the same height, return the root, else return the larger height subtree's root/sub-root.
def subtreeWithAllDeepest(self, root: TreeNode) -> TreeNode:
    return self.get_depth(root)[1]

#return [depth, node_with_subtree_having_largest_depth]
#The subtree of a node is that node, plus the set of all descendants of that node. 
def get_depth(self, root):
    if not root: return [0, None]
    left, right = self.get_depth(root.left), self.get_depth(root.right)
    if left[0] > right[0]: return [left[0]+1, left[1]]
    elif left[0] < right[0]: return [right[0]+1, right[1]]
    else: return [left[0]+1, root]



# 951
# some flipped some are not.
def flipEquiv(self, root1: TreeNode, root2: TreeNode) -> bool:
    if not root1 or not root2: return root1 == root2
    if root1.val != root2.val: return False
    return (self.flipEquiv(root1.left, root2.right) and self.flipEquiv(root1.right, root2.left)) or \
    (self.flipEquiv(root1.left, root2.left) and self.flipEquiv(root1.right, root2.right))



# 776
# trick: [node1, node2] node1 is a subtree containing values <= V, and node2 has values > V
# subtrees are still bst, so it will return [small, large] back just utilize the return nodes
def splitBST(self, root: TreeNode, V: int) -> List[TreeNode]:
    res = [None, None]
    if not root: return res
    
    if V >= root.val:
        res[0] = root
        small, large = self.splitBST(root.right, V)
        root.right = small
        res[1] = large
    else:
        res[1] = root
        small, large = self.splitBST(root.left, V)
        root.left = large
        res[0] = small
    return res
        



# 245
def longestConsecutive(self, root: TreeNode) -> int:
        if not root: return 0
        self.max_cnt = 0
        self.dfs(root, 0, root.val)
        return self.max_cnt
    

def dfs(self, root, cnt, target):
    if not root: return 0
    cnt = 1 + (cnt if root.val == target else 0)
    self.max_cnt = max(self.max_cnt, cnt)
    self.dfs(root.left, cnt, root.val+1) 
    self.dfs(root.right, cnt, root.val+1)
        


# 1145
# trick: take the largest subtree of first player, if it is greater than half of the total nodes, second player will win.
# becasue player cannot go across other player's node. each player can only visited neigbours, left , right, parent.
def btreeGameWinningMove(self, root: TreeNode, n: int, x: int) -> bool:
    count = [0, 0]
    def count_nodes(root):
        if not root: return 0 
        l, r = count_nodes(root.left), count_nodes(root.right)
        if root.val == x:
            count[0], count[1] = l, r
        return l + r + 1 
    t = count_nodes(root)
    return (t//2) < max(max(count), n - sum(count)-1)
        


# 971
def __init__(self):
        self.i = 0
    
def flipMatchVoyage(self, root: TreeNode, voyage: List[int]) -> List[int]:
    res = []
    return res if self.dfs(root, voyage, res) else [-1]
    
#if can flip given tree to the target tree
def dfs(self, root, voyage, res):
    if not root: return True
    if root.val != voyage[self.i]: return False
    self.i += 1
    if root.left and root.left.val != voyage[self.i]:
        res.append(root.val)
        root.left, root.right = root.right, root.left
    
    return self.dfs(root.left, voyage, res) and self.dfs(root.right, voyage, res)



# 508 
# subtree sum
def findFrequentTreeSum(self, root: TreeNode) -> List[int]:
        if not root: return []
        dic = collections.defaultdict(int)
        self.post_order(root, dic)
        n = max(dic.values())
        return [k for k, v in dic.items() if v == n]
        
        
def post_order(self, root, dic):
    if not root: return 0
    total = self.post_order(root.left, dic) + self.post_order(root.right, dic) + root.val
    dic[total] += 1
    return total



# 1104
def pathInZigZagTree(self, label: int) -> List[int]:
    level, node_cnt = 0, 1 
    ans = []
    while label >= node_cnt:
        node_cnt *= 2
        level += 1
        
    while label != 0:
        ans.append(label)
        mx, mi = 2**level - 1, 2**(level-1)
        #trick: to find label's parent node, find the distance from label to max (mx-label), and find the another label is (mx-label) away from mi. 
        # this label's parent become the label's parent due to the zigzag patern             
        label = (mi+(mx-label))//2
        level -=1 
    return ans[::-1]


        
# 1110
# good example of recursively pass down info by argument, and pass up by return value at the same time.
def delNodes(self, root: TreeNode, to_delete: List[int]) -> List[TreeNode]:
    res = []
    self.findComponnents(root, False, res, set(to_delete))
    return res


# alogrithm: 
# if a node does not need to be deleted and has no parent, add to result, has_parent is true
# if a node needs to be deleted, set has_parent to false 
def findComponnents(self, root, has_parent, res, to_delete):
    if not root: return None
    
    if root.val not in to_delete:
        if not has_parent: res.append(root)
        root.left = self.findComponnents(root.left, True, res, to_delete)
        root.right = self.findComponnents(root.right, True, res, to_delete)
        return root
    else:
        root.left = self.findComponnents(root.left, False, res, to_delete)
        root.right = self.findComponnents(root.right, False, res, to_delete)
        return None 



# 814
# postorder
def pruneTree(self, root: TreeNode) -> TreeNode:
    if not root: return None
    root.left = self.pruneTree(root.left)
    root.right = self.pruneTree(root.right)
    if not root.left and not root.right and root.val != 1: return None
    return root 


    
# 366
def findLeaves(self, root: TreeNode) -> List[List[int]]:
        res = []
        self.height(root, res)
        return res
    

# trick: bottom up as long as nodes with same height, they can be collected together, you dont really need to remove the nodes. 
def height(self, root, res): 
    if not root: return -1 
    
    h = 1 + max(self.height(root.left,res), self.height(root.right, res))
    
    if len(res) == h: res.append([])
    res[h].append(root.val)
    return h
        



# 117
def connect_II(self, root: 'Node') -> 'Node':
    if not root: return root
    
    queue = collections.deque([root])
    while queue:
        size = len(queue)
        for i in range(size):
            front = queue.popleft()
            # last node should be point to none
            if i < size - 1:
                front.next = queue[0]
            
            if front.left: queue.append(front.left)
            if front.right: queue.append(front.right)
    return root
        

# 652 
# tip serialize subtrees 
def findDuplicateSubtrees(self, root: TreeNode) -> List[TreeNode]:
    res = []
    self.serialize(root, {}, res)
    return res


def serialize(self, root, dic, res):
    if not root: return '#'
    serial = str(root.val) + ',' + self.serialize(root.left, dic, res) + ',' + self.serialize(root.right, dic, res)
    if serial in dic and dic[serial] == 1: res.append(root)
    dic[serial] = dic.get(serial, 0) + 1
    return serial
        


# 116
# tip: must start from top down or preorder.
def connect(self, root: 'Node') -> 'Node':
    if not root: return None
    if root.left:
        # connect children 
        root.left.next = root.right
        # connect gap of grandchildren
        if root.next: 
            root.right.next = root.next.left
    
    self.connect(root.left)
    self.connect(root.right)
    return root


# iter
# build the bridge then connecting the gaps.
def connect(self, root: 'Node') -> 'Node':
    r = root
    while r and r.left:
        cur = r
        while cur:
            cur.left.next = cur.right
            cur.right.next = cur.next if not cur.next else cur.next.left
            cur = cur.next
        r = r.left
    return root


# 889
def constructFromPrePost(self, pre: List[int], post: List[int]) -> TreeNode:
    return self.build(pre, post, 0, 0, len(pre)-1)


def build(self, pre, post, i, left, right):
    if left > right: return None 
    if left == right: return TreeNode(pre[left])
    
    root = TreeNode(pre[i])
    right_node_idx = pre.index(post[post.index(pre[i])-1])

    root.left = self.build(pre, post, i+1, i+1, right_node_idx-1)
    root.right = self.build(pre, post, right_node_idx, right_node_idx, right)
    
    return root


# 106
# careful calculating the size of subtree 
def buildTree(self, inorder: List[int], postorder: List[int]) -> TreeNode:
    return self.build(len(postorder)-1, 0, len(inorder)-1, inorder, postorder)
    
    
def build(self, r, i, j, inorder, postorder):
    if  i > j : return 
    idx = inorder.index(postorder[r])
    root = TreeNode(postorder[r])
    
    root.right = self.build(r-1, idx+1, j, inorder, postorder)
    root.left = self.build(r-(j-idx)-1, i, idx-1,inorder, postorder)
    return root
        


# 250
def countUnivalSubtrees(self, root: TreeNode) -> int:
    self.count = 0
    self.check(root)
    return self.count


def check(self, root):
    if not root:return True 
    
    left, right = self.check(root.left), self.check(root.right)
    if left and right and (not root.left or root.left.val == root.val) and (not root.right or root.val == root.right.val): 
        self.count += 1 
        return True 
    return False



# 285
# maintain a prev variable
def inorderSuccessor(self, root, p):
    stack = []
    prev = None
    while stack or root:
        while root:
            stack.append(root)
            root = root.left
        
        if stack:
            root = stack.pop()
            if prev == p.val: return root
            prev = root.val 
            root = root.right
    return None



# 314
# Given a binary tree, return the vertical order traversal of its nodes' values. (ie, from top to bottom, column by column).
# If two nodes are in the same row and column, the order should be from left to right.
# Examples 1:
# Input: [3,9,20,null,null,15,7]

#    3
#   /\
#  /  \
#  9  20
#     /\
#    /  \
#   15   7 

# Output:
# [
#   [9],
#   [3,15],
#   [20],
#   [7]
# ]
# with sorting
# trick: root at col 0 and row 0, do a bfs update each node's col and store in dic's same col list.
def verticalOrder(self, root):
    col_dic = collections.defaultdict(list)
    queue = collections.deque()
    queue.append((root, 0))
    
    while queue:
        node, col = queue.popleft()
        if node:
            col_dic[col].append(node.val)
            queue.append((node.left, col-1))
            queue.append((node.right, col+1))
            
    return [col_dic[k] for k in sorted(col_dic.keys())]


# without sorting
# use two variable to indicate the col range
def verticalOrder(self, root):
    if not root: return []
    col_dic = collections.defaultdict(list)
    queue = collections.deque()
    queue.append((root, 0))
    mi, mx = 0, 0 
    while queue:
        node, col = queue.popleft()
        if node:
            col_dic[col].append(node.val)
            mi = min(mi, col)
            mx = max(mx, col)
            queue.append((node.left, col-1))
            queue.append((node.right, col+1))
            
    return [col_dic[k] for k in range(mi, mx+1)] 




# 589
def preorder(self, root):
        if not root:
            return []
        res = []
        self.preorder_helper(root, res)
        return res


def preorder_helper(self, r, res):
    res.append(r.val)
    for n in r.children:
        self.preorder_helper(n, res)

# 590


def postorder(self, root):
    res = []
    if not root:
        return []
    self.postorder_helper(root, res)
    return res


def postorder_helper(self, r, res):
    for n in r.children:
        self.postorder_helper(n, res)
    res.append(r.val)


# 94
def inorderTraversal(self, root):
    stack, res = [], []
    while stack or root:
        while root:
            stack.append(root)
            root = root.left

        if stack:
            root = stack.pop()
             # process the node value here. 
            res.append(root.val)
            root = root.right
    return res


# 144
# cleaner version
def preorderTraversal(self, root):
    stack, ans = [], []
    while stack or root:
        while root:
            # process the node value here. 
            ans.append(root.val)
            stack.append(root)
            root = root.left
        
        if stack:
            root = stack.pop()
            root = root.right
    return ans


# alternative
def preorderTraversal(self, root):
    stack, res = [root],  []
    if not root:
        return res
    while stack:
        root = stack.pop()
        res.append(root.val)
        if root.right:
            stack.append(root.right)
        if root.left:
            stack.append(root.left)
    return res


# 417 
# thinking backward or reversly is key to solve this problem.
# trick: do dfs start from all four edges positions. 
def pacificAtlantic(self, matrix: List[List[int]]) -> List[List[int]]:
        res = []
        if not matrix: return res
        self.directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        n, m = len(matrix), len(matrix[0])
        # maintain two tables to avoid doing dfs for every position to avoid unnecessary search.
        p_visited = [[False for _ in range(m)] for _ in range(n)]
        a_visited = [[False for _ in range(m)] for _ in range(n)]
        #left and right edges
        for i in range(n):
            self.dfs(matrix, i, 0, p_visited)
            self.dfs(matrix, i, m-1, a_visited)
        #up and down edges
        for j in range(m):
            self.dfs(matrix, 0, j, p_visited)
            self.dfs(matrix, n-1, j, a_visited)
        
        for a in range(n):
            for b in range(m):
                if p_visited[a][b] and a_visited[a][b]:
                    res.append([a, b])
        return res
    
    
# normal dfs template with little modification. 
def dfs(self, matrix, i, j, visited):
    visited[i][j] = True
    for x, y in self.directions:
        x, y = x+i, y+j
        if 0 <= x < len(matrix) and 0 <= y < len(matrix[0]) and not visited[x][y] and matrix[x][y] >= matrix[i][j]:
            self.dfs(matrix, x, y, visited)


# 130
# dfs
def solve(self, board: List[List[str]]) -> None:
    if not board: return 
    starts = []
    n, m = len(board), len(board[0])
    for i in range(n):
        for j in range(m):
            if (i == 0 or i==n-1 or j == 0 or j == m-1) and board[i][j] == 'O':
                starts.append((i, j))
    visited = set()
    for pos in starts:
        if (pos[0], pos[1]) not in visited:
            self.dfs(board, pos, visited)
    
    for i in range(n):
        for j in range(m):
            if (i, j) not in visited and board[i][j] == 'O':
                board[i][j] = 'X'
        
        
def dfs(self, board, pos, visited):
    if board[pos[0]][pos[1]] == 'O' and (pos[0], pos[1]) not in visited:
        visited.add((pos[0], pos[1]))
        for x, y in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            xx, yy = x + pos[0], y + pos[1]
            if 0 <= xx < len(board) and 0 <= yy < len(board[0]):
                self.dfs(board, [xx, yy], visited)


# 200
class Solution(object):
    def numIslands(self, grid):
        if not grid:
            return 0
        n, m = len(grid), len(grid[0])
        uf = Union_Find(grid)

        for i in range(n):
            for j in range(m):
                if grid[i][j] == '1':
                    for x, y in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                        x += i
                        y += j
                        if 0 <= x < n and 0 <= y < m:
                            n1, n2 = (i * m + j), (x * m + y)
                            if not uf.connected(n1, n2) and grid[x][y] == '1':
                                uf.union(n1, n2)
        return uf.count

# tip: root's parent is itself
class Union_Find(object):
    def __init__(self, grid):
        n, m = len(grid), len(grid[0])
        self.id = [0] * n * m
        self.size = [1] * n * m
        self.count = 0
        for i in range(n):
            for j in range(m):
                if grid[i][j] == '1':
                    id = i * m + j
                    self.id[id] = id
                    """ only count components with 1 in the grid """
                    self.count += 1

    def connected(self, p, q):
        return self.find(p) == self.find(q)


    def union(self, p, q):
        p_root = self.find(p)
        q_root = self.find(q)

        if self.size[p_root] > self.size[q_root]:
            self.id[q_root] = p_root
            self.size[p_root] += self.size[q_root]
        else:
            self.id[q_root] = p_root
            self.size[q_root] += self.size[p_root]
        # reduce component with 1s by 1
        self.count -= 1


    """ find root """
    def find(self, p):
        while self.id[p] != p:
            self.id[p] = self.id[self.id[p]]
            p = self.id[p]
        return p



# union find without UF class
def numIslands(self, grid: List[List[str]]) -> int:
    if not grid: return 0 
    n, m = len(grid), len(grid[0])
    # create the id array and initialize it
    ID = [0] * n * m
    component_size = [0] * n * m
    components_cnt = 0
    for i in range(n):
        for j in range(m):
            if grid[i][j] == '1':
                pos = i * m + j
                ID[pos] = pos # this '1' itself is a component so it is root of itself
                component_size[pos] = 1
                components_cnt += 1
    
    for i in range(n):
        for j in range(m):
            if grid[i][j] == '1':
                for x, y in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                    x , y = x+i, y+j
                    if 0 <= x < n and 0 <= y < m and grid[x][y] == '1':
                        cur, nei = i*m+j, x*m+y
                        if self.find(cur, ID) != self.find(nei, ID):
                            self.union(cur, nei, ID, component_size)
                            components_cnt -= 1
    return components_cnt


def union(self, p, q, ID, size):
    p_root = self.find(p, ID)
    q_root = self.find(q, ID)
    
    if size[p_root] > size[q_root]:# optional but optimized
        ID[q_root] = p_root
        size[p_root] += size[q_root]
    else:
        ID[p_root] = q_root
        size[q_root] += size[p_root]
    


def find(self, p, ID):
    while p != ID[p]:
        # get inside here meaning that p's parent is not the root. 
        ID[p] = ID[ID[p]]# compression, optional but optimized
        p = ID[p]# looking for p's root, is same as looking for p's grandparent's root
    return p 



# bfs
# set 0 right after append to queue.
def numIslands(self, grid: List[List[str]]) -> int:
        if not grid: return 0
        self.directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        n, m = len(grid), len(grid[0])
        cnt = 0
        for i in range(n):
            for j in range(m):
                if grid[i][j] == '1':
                    self.bfs(i, j, grid)
                    cnt += 1
        return cnt
        
# key: queue stores all the positions that are visited.  
# reason: if you store all the position unvisited, will cause duplicates being added into queue during append operation.
# bfs: the same level of nodes should be all visited before adding next level to the queue.
def bfs(self, i, j, grid):
    queue = collections.deque([(i, j)])
    grid[i][j] = '0'
    while queue:
        a, b = queue.popleft()
        for x, y in self.directions:
            if 0 <= x+a < len(grid) and 0 <= y+b < len(grid[0]) and grid[x+a][y+b] == '1':
                grid[x+a][y+b] = '0'
                queue.append((x+a, y+b))
    


# 694
def numDistinctIslands(self, grid: List[List[int]]) -> int:
    islands = set()
    for i in range(len(grid)):
        for j in range(len(grid[i])):
            path = []
            if grid[i][j] == 1: 
                self.dfs(grid, i, j, path, 'b') # mark begining
                islands.add(''.join(path))
    return len(islands)

# key: serialize the path for starting at given position, marked as begin of the search, then once four directions are visited, mark end. 
#  must mark ending, else the shape of islands may differnt, which could end up having same serilization.               
def dfs(self, grid, i, j, path, d):
    if i < 0 or i >= len(grid) or j < 0 or j >= len(grid[0]) or grid[i][j] != 1:
        return
    
    grid[i][j] = 0
    path.append(d)
    self.dfs(grid, i+1, j, path, 'd')
    self.dfs(grid, i-1, j, path, 'u')
    self.dfs(grid, i, j+1, path, 'r')
    self.dfs(grid, i, j-1, path, 'l')
    path.append('e') # mark end
                

# 399  
# union find -> difficult
class Solution(object):
    def calcEquation(self, equations, values, queries):
        roots = {}  # the root of each char
        dist = {}  # a -> root, a/f -> r, where f is the distance
        res = []

        # initialize the components
        for i in range(len(equations)):
            x, y = equations[i]
            factor = values[i]
            if x not in roots:
                roots[x] = x
                dist[x] = 1.0
            if y not in roots:
                roots[y] = y
                dist[y] = 1.0
            self.union(roots, dist, x, y, factor)

        for p, q in queries:
            """ tip: do not write this conditoon as (p or q) not in roots, this is differnt. the expression inside the parenthesis will evaluated first, than the rest.  """
            if p not in roots or q not in roots:
                res.append(-1.0)
            else:
                if self.find(roots, dist, p) != self.find(roots, dist, q):
                    res.append(-1.0)
                else:
                    # distance from p to root divides distance from q to root will get distance from p to q, which is p / q
                    res.append(dist[p]/dist[q])
        return res

    # path compression: make the found root as the parent of current node
    # the idea is to flattern the component, so next time revisting the same node, it does not have to traverse intermediate nodes again
    def find(self, r, d, p):
        if r[p] == p:
            return p
        tmp = r[p]
        r[p] = self.find(r, d, r[p])
        # accumulate the distance on the current node, this is why we use recurive version of compression, top-down
        d[p] *= d[tmp]
        return r[p]

    def union(self, r, d, p, q, f):
        p_r = self.find(r, d, p)
        q_r = self.find(r, d, q)
        if p_r != q_r:
            r[p_r] = q_r
            """ p_r to q_r distance is calculated by following formula, a little bit math   """
            d[p_r] = f * (d[q]/d[p])


""" bfs solution """
# find the shortest path from q[0] -> q[1], thus BFS 
def calcEquation(self, equations: List[List[str]], values: List[float], queries: List[List[str]]) -> List[float]:
    # build the division graph, start -> (dist, end)
    graph = collections.defaultdict(list)
    for i, eq in enumerate(equations):
        graph[eq[0]].append((values[i], eq[1]))
        graph[eq[1]].append((float(1/values[i]), eq[0]))
    
    res = []
    
    # processs queries
    for start, end in queries:
        # if either not in graph return -1
        if start not in graph or end not in graph:
            res.append(-1)
        elif start == end: 
            res.append(1)
        else:
            res.append(self.bfs(start, end, graph, set()))
    return res

# find the shortest path from start to end, calculate the product of the weight of the paths.
def bfs(self, start, end, graph, visited):
    queue = collections.deque([(1, start)])
    while queue:
        size = len(queue)
        while size: 
            w, cur = queue.popleft()
            visited.add(cur)
            if cur == end: return w
            for d, nei in graph[cur]:
                if nei not in visited:
                    queue.append((w*d, nei))
            size -= 1
    return -1



# tree 
class Node(object):
    def __init__(self, val=None, children=None):
        self.val = val
        self.children = children


class Solution(object):
    def levelOrder(self, root):
        if not root:
            return []
        import collections
        queue = collections.deque()
        res = []
        queue.append(root)
        while queue:
            size = len(queue)
            tmp = []
            while size:
                front = queue.popleft()
                for c in front.children:
                    queue.append(c)
                tmp.append(front.val)
                size -= 1
            res.append(tmp)
        return res



# 101
def isSymmetric(self, root):
        return self.helper(root, root)


def helper(self, r1, r2):
    # the order of two condition below cannot be reversed
    if not r1 and not r2:
        return True
    if not r1 or not r2:
        return False
    return r1.val == r2.val and self.helper(r1.right, r2.left) and self.helper(r1.left, r2.right)


def isSymmetric_iter(self, root):
    import collections
    # intuition: mirror of itself, we can actually compare two tree at the same time
    queue = collections.deque([root, root])
    while queue:
        n1 = queue.popleft()
        n2 = queue.popleft()
        if not n1 and not n2:
            continue
        if not n1 or not n2:
            return False
        if n1.val != n2.val:
            return False
        queue.append(n1.right)
        queue.append(n2.left)
        queue.append(n1.left)
        queue.append(n2.right)
    return True


# dfs
def maxDepth(self, root):
    if not root:
        return 0
    return max(self.maxDepth(root.left), self.maxDepth(root.right)) + 1

# find number of level will be the depth of the tree


def maxDepth_bfs(self, root):
    if not root:
        return 0
    import collections
    queue = collections.deque()
    queue.append(root)
    level = 0
    while queue:
        size = len(queue)
        while size:
            front = queue.popleft()
            if front.left:
                queue.append(front.left)
            if front.right:
                queue.append(front.right)
            size -= 1
        level += 1
    return level


# very slow coz repeatedly calculate the height of nodes from bottom up
def isBalanced(self, root):
    if not root:
        return True
    return self.isBalanced(root.left) and self.isBalanced(root.right) and abs(self.height(root.left) - self.height(root.right)) <= 1


def height(self, root):
    if not root:
        return 0
    return max(self.height(root.left), self.height(root.right)) + 1

#  faster version: calculate the height and at the same time check if subtree is
# follow the balanced tree definition by using a signal -1 to indicate that.


def isBalanced(self, root):
    def check(root):
        if not root:
            return 0
        left = check(root.left)
        right = check(root.right)
        if left == -1 or right == -1 or abs(left - right) > 1:
            return -1
        return max(left, right) + 1
    return check(root) != -1


# 501
# key: find the most frequent elements in a sorted array with duplicates in one pass using O(1) space. 
# tip: see bst, think it as a sorted array. 
def findMode(self, root: TreeNode) -> List[int]:
        self.max_count = 0
        self.cur_count = 0
        self.prev = None
        self.res = []
        self.inorder(root)
        return self.res
    
    
def inorder(self, root):
    if not root: return
    self.inorder(root.left)
    self.cur_count = 1 if root.val != self.prev else self.cur_count + 1
    if self.cur_count == self.max_count:
        self.res.append(root.val)
    elif self.cur_count > self.max_count:
        self.res = [root.val]
        self.max_count = self.cur_count
    self.prev = root.val
    self.inorder(root.right)



# 530
# similar as above
def getMinimumDifference(self, root: TreeNode) -> int:
        self.min_diff = float('inf')
        self.prev = None 
        self.inorder(root)
        return self.min_diff
    
    
def inorder(self, root):
    if not root: return 
    
    self.inorder(root.left)
    if self.prev != None:
        self.min_diff = min(abs(self.prev - root.val), self.min_diff)
    self.prev = root.val   
    self.inorder(root.right)


# 897 
# key: for root, convert root.right into a linkedlist, and insert this list in between root and root's parent. 
# thus, we need to maintain two states, (root, parent/tail)
def increasingBST(self, root: TreeNode) -> TreeNode:
    return self.inorder(root, None)
        

def inorder(self, root, tail):
    if not root: return tail
    head = self.inorder(root.left, root)
    root.left = None
    # insert right subtree list in between root and tail
    root.right = self.inorder(root.right, tail)
    return head
        
        


# 111
def minDepth(self, root):
        if not root: return 0
        left = self.minDepth(root.left)
        right = self.minDepth(root.right)
        if left == 0:
            return right + 1
        if right == 0:
            return left + 1
        return min(left, right) + 1


def deepestLeavesSum(self, root):
    if not root:
        return 0
    queue = collections.deque([root])
    res = None
    while queue:
        size = len(queue)
        res = 0
        while size:
            front = queue.popleft()
            res += front.val
            if front.left:
                queue.append(front.left)
            if front.right:
                queue.append(front.right)
            size -= 1
    return res


# using set
def isUnivalTree(self, root):
    def helper(root, path):
        if not root:
            return
        path.add(root.val)
        helper(root.left, path)
        helper(root.right, path)

    path = set()
    helper(root, path)
    return len(path) == 1

# O(1) space


def isUnivalTree_(self, root):
    if not root:
        return True
    if root.left:
        if root.val != root.left.val:
            return False
    if root.right:
        if root.val != root.right.val:
            return False
    return self.isUnivalTree(root.left) and self.isUnivalTree(root.right)




def isSubtree(self, s, t):
        if not s: return False
        if self.is_same(s, t):
            return True
        return self.isSubtree(s.left, t) or self.isSubtree(s.right, t)


def is_same(self, r1, r2):
    if not r1 and not r2: return True 
    if not r1 or not r2: return False 
    return r1.val == r2.val and self.is_same(r1.left, r2.left) and self.is_same(r1.right, r2.right)
    




def levelOrder(self, root):
    res = []
    if not root: return res
    self.level_dfs(root, 0, res)
    return res
        
# dfs way of level order traversal is actually a preorder traversal with addtional condition to create the list and adding the same level result to that list
def level_dfs(self, root, level, res):
    if not root: return
    # when level == to size of res meaning the lists in the res are not enough coz on the next line we need to use the level to find the right list to append current node
    # or we at at new level, so we need a new list to hold that level's items.
    if level == len(res):
        res.append([])
    """ preorder traversal """
    res[level].append(root.val)
    self.level_dfs(root.left, level+1, res)
    self.level_dfs(root.right, level+1, res)
    
    
# 538
# root.val += rightside total
def __init__(self):
    self.right_side = 0 

def convertBST(self, root: TreeNode) -> TreeNode:
    if not root: return None 
    root.right = self.convertBST(root.right)
    root.val += self.right_side
    self.right_side = root.val
    root.left = self.convertBST(root.left)
    return root
 



# 107 reverse level order 
def levelOrderBottom(self, root):
    res = []
    if not root: return res
    self.dfs(root, 0, res)
    return res
    

def dfs(self, root, level, res):
    if not root:return
    if level == len(res):
        res.insert(0, [])
        
    res[len(res)-level-1].append(root.val)
    self.dfs(root.left, level+1, res)
    self.dfs(root.right, level+1, res)


def hasPathSum(self, root, sum):
    if not root: return False
    if not root.left and not root.right and (sum - root.val == 0):
        return True
    return self.hasPathSum(root.left, sum-root.val) or self.hasPathSum(root.right, sum-root.val)
        

# 257 
def binaryTreePaths(self, root):
    res = []
    self.dfs(root, '', res)
    return res

def dfs(self, root, path, res):
    if not root: return
    path += str(root.val) #update path here can skip many edge cases.
    if not root.left and not root.right:
        res.append(path)
    self.dfs(root.left, path + '->', res)
    self.dfs(root.right, path + '->', res)



# 113
def pathSum(self, root, sum):
    res = []
    self.helper(root, [], res , sum)
    return res 
    
def helper(self, root, path, res, total):
    if not root: return
    if not root.left and not root.right and (total - root.val == 0):
        path.append(root.val)
        res.append(path)
        return
    
    self.helper(root.left, path + [root.val], res, total - root.val)
    self.helper(root.right, path + [root.val], res, total - root.val)


# 437 
""" brute force solution try all the nodes in the tree to find all the paths """
class Solution(object):
    def __init__(self):
        self.count = 0

    def pathSum(self, root, sum):
        self.helper(root, sum)
        return self.count

    def helper(self, root, s):
        if not root:
            return
        self.num_path(root, s)
        self.helper(root.left, s)
        self.helper(root.right, s)

    def num_path(self, root, s):
        if not root:
            return
        """ dont return the stack frame coz it will have edge case, for example [1,-2,-3,1,3,-2,null,-1], -1 
        for this case, it will not include path [1 -> -2 -> 1 -> -1] 
        if not returning, the recurive call will go deeper to include this path """
        if s == root.val:
            self.count += 1

        self.num_path(root.left, s - root.val)
        self.num_path(root.right, s - root.val)



""" prefix sum solution """
def pathSum(self, root: TreeNode, total: int) -> int:
    self.cnt = 0
    dic = {0: 1}
    self.search(dic, 0, root, total)
    return self.cnt
    

def search(self, dic, pre_sum, root, total): 
    if not root: return
    cur = pre_sum + root.val
    if cur - total in dic:
        self.cnt += dic[cur-total]
    dic[cur] = dic.get(cur, 0) + 1 #prefix sum at root(inclusive)
    # search any path of subtrees of current root
    self.search(dic, cur, root.left, total)
    self.search(dic, cur, root.right, total)
    # prefix sum at cur is done, can not be used anymore, thus clear it.
    dic[cur] -= 1 



# 236
""" single stack frame logic: go find the p or q node from left or right side of tree, and return the results(references) to p or q or not found
if result return inlcude both references of p or q meaning current root is the LCA else if p is found, p is the parent of q return p else return q """
def lowestCommonAncestor(self, root, p, q):
    if not root: return None
    # similar to path compression, return the found root reference to the above level
    if root in (p, q): return root
    """ recursion, each stack frame will maintain or remember some states and waiting for rest of recursion code to return and then combine the result to solve the problem """
    left = self.lowestCommonAncestor(root.left, p, q)
    right = self.lowestCommonAncestor(root.right, p, q)

    return root if (left and right) else (left or right)


# iterative with stack and setup the child to parent mappings
def lowestCommonAncestor_iter(self, r, p, q):
    parent = {r: None}
    stack = [r]
    while p not in parent or q not in parent:
        top = stack.pop()
        if top.left:
            stack.append(top.left)
            parent[top.left] = top
        if top.right:
            stack.append(top.right)
            parent[top.right] = top
    p_ancestors = set()
    while p:
        p_ancestors.add(p)
        p = parent[p]
    
    while q not in p_ancestors:
        q = parent[q]
    
    return q





# 669 
# define what this function will do. 
# this function will trim the BST  and return the new root of the tree.
def trimBST(self, root, L, R):
    if not root: return 
    """ basically this is preorder traversal. 
    check if root val is in range 
    if root is not in range, starts function at its children using the BST characteristics """
    # if root is not in range, we only need to check either left or right child
    if root.val < L: return self.trimBST(root.right, L, R)
    if root.val > R: return self.trimBST(root.left, L, R)
    # if root is in range, we need to trim both children
    root.left = self.trimBST(root.left, L, R)
    root.right = self.trimBST(root.right, L, R)
    return root


# 814 
""" pre-order version """
def pruneTree_pre(self, root):
    if not root: return 
    if self.has_one(root): 
        root.left = self.pruneTree(root.left)
        root.right = self.pruneTree(root.right)
    else: 
        root = None
    return root 
    
    
def has_one(self, r):
    if not r: return False
    return r.val == 1 or self.has_one(r.left) or self.has_one(r.right)
    
""" post order, this will avoid using second helper function to check if subtree including 1's or not """
""" single stack logic: prune left subtree and prune right subtree, check if 1's in left or right subtree and if current root value is 1 or not
if no 1 found return None else return current root """
def pruneTree_post(self, root):
    if not root: return None 
    root.left = self.pruneTree(root.left)
    root.right = self.prunTree(root.right)
    if not root.left and not root.right and not root.val: return None
    return root


# 872
def leafSimilar(self, root1, root2):
    #compare if two nodes have same leaves
    return self.find_leaf(root1) == self.find_leaf(root2)

#return a list of leaves
def find_leaf(self, r):
    # if null tree, has no leaf, return empty list 
    if not r: return []
    # if a tree node has not children, that node is the leaf, add to the list and return
    if not r.left and not r.right: return [r.val]
    #combine leaves from left and right subtrees
    return self.find_leaf(r.left) + self.find_leaf(r.right)
    
# 543
# very slow to-down preorder traversal, for every node need to calculate the height of left and right subtree
def diameterOfBinaryTree(self, root):
    if not root:
        return 0
    left = self.diameterOfBinaryTree(root.left)
    right = self.diameterOfBinaryTree(root.right)
    thru_root = self.height(root.left) + self.height(root.right)
    return max(thru_root, left, right)

# this is an postorder traversal, bottom up

def height(self, r):
    if not r:
        return 0
    return max(self.height(r.left), self.height(r.right)) + 1


# some optimalization: using dictionary to precalculate the heights of each node
""" tip: number of nodes in the subtrees equal to the total edges count """


def diameterOfBinaryTree_dic(self, root):
    dic = {}
    self.height(root, dic)
    return self.helper(root, dic)


def helper(self, root, dic):
    if not root:
        return 0
    left = self.helper(root.left, dic)
    right = self.helper(root.right, dic)
    left_height = dic[root.left] if root.left else 0
    right_height = dic[root.right] if root.right else 0
    thru_root = left_height + right_height
    return max(thru_root, left, right)


def height(self, r, dic):
    if not r:
        return 0
    h = max(self.height(r.left, dic), self.height(r.right, dic)) + 1
    dic[r] = h
    return h


# more optimization: maintain a global variable to keep a max distance thru root.
class Solution(object):
    def __init__(self):
        self.ans = 0

    def diameterOfBinaryTree(self, root):
        self.height(root)
        return self.ans

    def height(self, r):
        if not r:
            return 0
        left, right = self.height(r.left), self.height(r.right)
        # left + right is total number of nodes in the subtrees , which is equal to the total edge count going thru root.
        self.ans = max(self.ans, left+right)
        return max(left, right) + 1


# 124
class Solution(object):
    def __init__(self):
        self.res = float('-inf')

    def maxPathSum(self, root):
        self.maxPathSumFromRoot(root)
        return self.res

    # return max path sum from given root node to some node in the tree
    """ find the path that has the largest sum from root to some node in the tree and return the sum """

    def maxPathSumFromRoot(self, root):
        if not root:
            return float('-inf')
        left = self.maxPathSumFromRoot(root.left)
        right = self.maxPathSumFromRoot(root.right)
        # this updates the largest path sum along the way back to the top roots
        '''the max path sum must be some path that go to or go thru some root in the tree'''
        self.res = max(self.res, max(left, 0) + max(right, 0) + root.val)
        # the returned is just one path starting from current root
        return max(left, right, 0) + root.val


""" tip: post-order our single stack call logic is better apply to the last stack that close to the base case because we are work from the bottom up """

# 687 similar to above problem
# bottom-up, if r.val != r.right.val, just treat r as a leaf, thus the height is 0. 
class Solution:
    def __init__(self):
        self.longest = 0
        

    def longestUnivaluePath(self, root: TreeNode) -> int:
        self.height(root)
        return self.longest
    

    def height(self, root):
        if not root: return -1
        left = self.height(root.left)
        right = self.height(root.right)
        left = (1 + left) if root.left and root.left.val == root.val else 0 # if not equal, treat the root node as the leaf, thus set height to 0 
        right = (1 + right) if root.right and root.right.val == root.val else 0
        self.longest = max(self.longest, left + right)
        return max(left, right) 



# 235
# if p or q is equal to value of root, that is the LCA
def lowestCommonAncestor(self, root, p, q):
    if not root:
        return
    if p.val < root.val and q.val < root.val:
        return self.lowestCommonAncestor(root.left, p, q)
    elif p.val > root.val and q.val > root.val:
        return self.lowestCommonAncestor(root.right, p, q)
    else:
        return root

# 1325
""" key is to recognize that we need to traverse from bottom to top, thus we use post-order """


def removeLeafNodes(self, root, target):
    if not root:
        return None
    """ dont put checking condition here , this will not work because it becomes preorder traversal"""
    root.left = self.removeLeafNodes(root.left, target)
    root.right = self.removeLeafNodes(root.right, target)
    # post-order, process the root at last.
    if not root.left and not root.right and root.val == target:
        return None
    else:
        return root


# 129
class Solution(object):
    def __init__(self):
        self.total = 0

    def sumNumbers(self, root):
        self.build_path(root, 0)
        return self.total

    def build_path(self, root, val):
        if not root:
            return
        if not root.left and not root.right:
            val = val*10+root.val
            self.total += val
            return

        self.build_path(root.left, val*10+root.val)
        self.build_path(root.right, val*10+root.val)



# 337
# recursive TLE
def rob_TLE(self, root):
    if not root:
        return 0
    if root:
        left = self.rob(root.left)
        right = self.rob(root.right)

    total = 0
    if root.left:
        total += self.rob(root.left.left)
        total += self.rob(root.left.right)
    if root.right:
        total += self.rob(root.right.left)
        total += self.rob(root.right.right)
    total += root.val
    return max(left + right, total)


# top down memo
def rob(self, root):
    dic = {}
    return self.helper(root, dic)


def helper(self, root, dic):
    if not root:
        return 0
    if root in dic:
        return dic[root]
    if root:
        left = self.helper(root.left, dic)
        right = self.helper(root.right, dic)

    total = 0
    if root.left:
        total += self.helper(root.left.left, dic)
        total += self.helper(root.left.right, dic)
    if root.right:
        total += self.helper(root.right.left, dic)
        total += self.helper(root.right.right, dic)
    total += root.val
    mx = max(left + right, total)
    dic[root] = mx
    return mx




# 979
# inorder traversal
# goal: how many moves needed for each subtree to be equally distributed? 
# key: if left and right subtrees are all ones, root must be one.
# root node maintains a balance of left and right subtrees deficit or overflow. update the val of root.
# number of moves for a tree r to be equally distributed with one coin equals to number of moves needed for left and right subtrees, plus the moves
# needed for making the root owning only one coin.
# note: this solution changed the root values of the tree
def distributeCoins(self, root: TreeNode) -> int:
    if not root: return 0
    left = self.distributeCoins(root.left) 
    root.val += (root.left.val-1 if root.left else 0)
    right = self.distributeCoins(root.right)
    root.val += (root.right.val-1 if root.right else 0)
    return left + right + abs(root.val-1)



# 501
def findMode_extra_space(self, root):
    if not root: return []
    dic = {}
    self.dfs(root, dic)

    max_freq = max(dic.values())

    res = []
    for v, f in dic.items():
        if f == max_freq: res.append(v)
    return res


def dfs(self, root, dic):
    if not root:
        return 
    self.dfs(root.left, dic)
    dic[root.val] = dic.get(root.val, 0) + 1 
    self.dfs(root.right, dic)


""" follow up implement O(1) space solution """
def findMode(self, root):
    pass 


class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

# 108
def sortedArrayToBST(self, nums):
    if nums:
        mid = len(nums)//2
        root = TreeNode(nums[mid])

        root.left = self.sortedArrayToBST(nums[:mid])
        root.right = self.sortedArrayToBST(nums[mid+1:])
        return root

# 530
""" key: BST doing a in order traversal will get sorted array. and the min absolute difference should exist in between two conitnuous number in the array
thus, we only need to maintain a prev variable to compare with current node and update the min_diff along the way """
class Solution(object):
    def __init__(self):
        self.min_diff = float('inf')
        self.prev = None

    # simply do a in-order traversal and return updated min_diff
    def getMinimumDifference(self, root):
        if not root:
            return self.min_diff

        self.getMinimumDifference(root.left)

        if self.prev:
            self.min_diff = min(self.min_diff, root.val - self.prev.val)

        self.prev = root

        self.getMinimumDifference(root.right)
        return self.min_diff



# 700
def searchBST(self, root, val):
    if not root: return     
    if root.val < val:
        return self.searchBST(root.right, val)
    elif root.val > val:
        return self.searchBST(root.left, val)
    else:
        return root
        


# 450
""" tip: when doing recursive problems, clearly define the what the function does and appy to each stack call see if it logically make sense """
# function deletes the node with key value, and return a new node that will maintain the BST property
""" copy_value implementation """
def deleteNode(self, root, key):
    if not root:
        return
    if root.val > key:
        root.left = self.deleteNode(root.left, key)
    elif root.val < key:
        root.right = self.deleteNode(root.right, key)
    else:
        if not root.left:
            return root.right
        elif not root.right:
            return root.left
        else:
            """ easiest way to maintain a BST when parent is deleted, is to find the min value on the right subtree and 
            put it at the parent position and remove it from its old position, thus both sides maintain bst 
            tip: all nodes on the right side are greater than the nodes on the left side, even the smallest on the right side.
             """
            min_node = self.find_min(root.right)
            #assign smallest value to the root and delete the node with smallest value will maintain left and right to be valid bst.
            root.val = min_node.val
            #this line will maintain the root.right to be a bst
            root.right = self.deleteNode(root.right, min_node.val)
    return root


# find the subtree on with min value
def find_min(self, root):
    while root.left:
        root = root.left
    return root


# problem does not say you cannot modify the tree. as long as the tree is bst, its ok .
def deleteNode(self, root: TreeNode, key: int) -> TreeNode:
        if not root: return 
        
        if root.val == key:
            if not root.left: return root.right
            if not root.right: return root.left
            return self.restructure(root)
        
        if key < root.val: 
            root.left = self.deleteNode(root.left, key)
        else:
            root.right = self.deleteNode(root.right, key)
        return root
    
# trick: simply add left subtree to the smallest node of right subtree's left subtree
def restructure(self, root):
    leftmost = root.right
    while leftmost.left: leftmost = leftmost.left
    leftmost.left = root.left
    return root.right

        



# 98 
# recursive solution
# preorder or postorder traversal both will work
def isValidBST(self, root):
    return self.isValidBST_helper(root, float('-inf'), float('inf'))


def isValidBST_helper(self, root, mi, mx):
    if not root: return True
    if root.val <= mi or root.val >= mx: return False

    left = self.isValidBST_helper(root.left, mi, root.val)
    right = self.isValidBST_helper(root.right, root.val, mx)

    return left and right

# iterative solution
""" using in-order traversal iterative way
because in-order traversal for bst will produce sorted array, we only need to maintain a prev, so if current node value is 
less and equal to prev, return false """

def isValidBST(self, root):
    stack, prev = [], None
    while stack or root:
        while root:
            stack.append(root)
            root = root.left
        root = stack.pop()
        """ tip: use is None or is not None as condition to avoid bugs """
        if prev is not None and root.val <= prev:
            return False
        prev = root.val
        root = root.right
    return True



# 230 
def kthSmallest(self, root, k):
    stack = []
    while stack or root:
        while root:
            stack.append(root)
            root = root.left
        
        root = stack.pop()
        k -= 1
        if k == 0: return root.val 
        root = root.right
    return -1
        


# 701 
# insert given node into the BST and return the modified BST 
def insertIntoBST(self, root, val):
    if not root: return TreeNode(val)
    if root.val > val:
        # insert into the leftsubtree and return the modified left subtree. 
        root.left = self.insertIntoBST(root.left, val)
        
    elif root.val < val:
        root.right = self.insertIntoBST(root.right, val)   
    return root



# 99 
# in-order iterative
def recoverTree_iter(self, r):
    stack, prev = [], None
    one, two = None, None
    while stack or r:
        while r:
            stack.append(r)
            r = r.left
        
        r = stack.pop()
        """ this part is a classic probelm: identify two swapped elements in sorted array """
        if prev and prev.val >= r.val:
            one = r
            if not two:
                two = prev
            else:
                break   
        """ end """
        prev = r
        r = r.right
    one.val, two.val = two.val, one.val

""" implement morris traversal in pace """
def recoverTree(self, r):
    pass 



# 968
def minCameraCover(self, root):
    """
    :type root: TreeNode
    :rtype: int
    """
    pass 




# 997 
# judge with n-1 in degrees and 0 out degree
# naive solution
def findJudge_naive(self, N, trust):
    if N == 1: return N
    graph = {}
    out_degree = {}
    for a, b in trust:
        graph[b] = graph.get(b, 0) + 1
        out_degree[a] = out_degree.get(a, 0) + 1

    for k, v in graph.items():
        if v == N-1:
            if k not in out_degree:
                return k
    return -1

""" clever solution """
# trick here is to think of the indegree and outdegree as + and -, and find the differences of each node
def findJudge(self, N, trust):
    count = [0] * (N+1)

    for a, b in trust:
        count[a] -= 1
        count[b] += 1

    for i in range(1, N+1):
        if count[i] == N-1:
            return i
    return -1


# greedy 
# two pointers, candidate: c
def findJudge(self, N: int, trust: List[List[int]]) -> int:
    # build trust graph 
    g = {}
    for a, b in trust:
        if a not in g:
            g[a] = set([b])
        else :
            g[a].add(b)

    c = 1 
    for i in range(1, N+1):
        if c in g and i in g[c]:
            c = i
    
    for j in range(1, N+1):
        if j == c: continue 
        if j not in g: return -1 
        elif c not in g[j]: 
            return -1 

    return c if c not in g else -1
        



# 863 
# my implementation. see more concise version below
# idea is the same: first convert the tree into a undirected graph and do a bfs search
import collections
def distanceK(self, root, target, K):
    res = []
    graph = collections.defaultdict(list)
    self.build_graph(root, graph)
    if len(graph.keys()) < K + 1: return res
    self.bfs(graph, target, res, K)
    return res
    
    
def build_graph_old(self, root, graph):
    if not root: return 
    if root.left: 
        graph[root].append(root.left)
        graph[root.left].append(root)
    if root.right:
        graph[root].append(root.right)
        graph[root.right].append(root)

    self.build_graph(root.left, graph)
    self.build_graph(root.right, graph)
    

def bfs(self, graph, src, res, K):
    queue = collections.deque()
    queue.append(src)
    level = 0
    visited = [False] * len(graph.keys())
    while queue:
        
        if level == K:
            while queue:
                res.append(queue.popleft().val) 
                
        size = len(queue)
        while size:
            front = queue.popleft()
            visited[front.val] = True
            for n in graph[front]:
                if not visited[n.val]:
                    queue.append(n)
            size -=1 
        level += 1



#another version of my implementaion
def distanceK(self, root, target, K):
    adj, res = collections.defaultdict(list), []
    self.build_graph(None, root, adj)
    queue = collections.deque()
    queue.append(target.val)
    visited = set()
    dist = 0
    while queue:
        size = len(queue)
        while size:
            front = queue.popleft()
            visited.add(front)
            if dist == K:
                res.append(front)
            for nei in adj[front]:
                if nei not in visited: 
                    queue.append(nei)
            size -= 1
        dist += 1
    return res
                    

""" lee's implementation conscise but a little slow """
def distanceK(self, root, target, K):
    graph = collections.defaultdict(list)
    self.build_graph(root, graph)
    bfs = [target.val]
    seen = set(bfs)
    for i in range(K):
        # for every node in bfs add its unseen neibhours to the new bfs list
        bfs = [y for x in bfs for y in graph[x] if y not in seen]
        seen |= set(bfs) #using set theory 'or' to combine two sets of elements
    return bfs


def build_graph_old(self, root, graph):
    if not root: return 
    if root.left: 
        graph[root.val].append(root.left.val)
        graph[root.left.val].append(root.val)
    if root.right:
        graph[root.val].append(root.right.val)
        graph[root.right.val].append(root.val)

    self.build_graph(root.left, graph)
    self.build_graph(root.right, graph)
    
""" cleaner version of building a graph """


def build_graph(self, parent, r, adj):
        if not r:
            return
        if r.left:
            adj[r.val].append(r.left.val)
        if r.right:
            adj[r.val].append(r.right.val)
        if parent:
            adj[r.val].append(parent.val)
        self.build_graph(r, r.left, adj)
        self.build_graph(r, r.right, adj)




# 785 
""" 
bipartite: for given graph, if all the nodes in the graph can be divided into two subsets, A and B, 
and for every edge in the graph, one of its node is in A, another is in B.
"""
# key:# the nei of a node cannot have the same color as the node itself.
# Acyclic graph will always be bipartite
def isBipartite(self, graph):
    # maintain color for each node.
    color = {}
    # graph maybe disconnected, meaning there are multiple sub-graphs might be bipartite
    for i in range(len(graph)):
        if i not in color:
            color[i] = 0
            if not self.check_color(graph, i, color):
                return False
    return True


def check_color(self, graph, src, color):
    for n in graph[src]:
        # this line will only execute when there is cycle. 
        if n in color:
            if color[n] == color[src]: 
                return False
        else:
            color[n] = 1 - color[src]
            if not self.check_color(graph, n, color):
                return False
    return True
            
    
    
# 332
# greedy + dfs. all tickets at least form a valid path.
# 1. must use all tickets. sometimes , if you do strick lexi order, there is no ticket to go back to where i came from. so the other tickets will not be used.
# 2. if possible, lexi order. 
# problem description is wrong, ex: [["JFK","LHR"] ,["JFK","MUC"]] => ["JFK","MUC","LHR"], I thought it would return [] 
# because no way you can use all the tickets because you cannot get back to JFK. 
def findItinerary(self, tickets):
    res = []
    import collections
    graph = collections.defaultdict(list)
    # sorted(tickets)[::-1] will ensure that when we pop item off, will in lexical order.
    for d, a in 3::-1]:
        graph[d].append(a)

    self.dfs(graph, 'JFK', res)
    return res[::-1]

# post order.
def dfs(self, graph, src, res):
    while graph[src]:
        self.dfs(graph, graph[src].pop(), res)
    res.append(src)


# another implementation using deque
def findItinerary(self, tickets: List[List[str]]) -> List[str]:
        #build graph
        g = collections.defaultdict(list)
        for start, dest in tickets: 
            g[start].append(dest)
        # map key to a queue
        graph = {}
        for k, alist in g.items():
            queue = collections.deque(sorted(alist))
            graph[k] = queue
        
        res = []
       # dfs to visit all the nodes
        self.dfs(graph, res, 'JFK')
        return ["JFK"] + res[::-1]


def dfs(self, graph, res, start):
    if start not in graph:return 
    while graph[start]:
        cur = graph[start].popleft()
        self.dfs(graph, res, cur)
        res.append(cur)
            

""" an Eulerian trail (or Eulerian path) is a trail in a finite graph that visits every edge exactly once (allowing for revisiting vertices). """
def findItinerary(self, tickets):
    tickets.sort()
    ticket = {}
    for depart, arrival in tickets:
        if depart not in ticket:
            tmp = collections.deque()
            ticket[depart] = tmp
        ticket[depart].append(arrival)

    res = []
    self.build('JFK', ticket, res)
    return res[::-1]

# post-order traversal
def build(self, src, ticket, res): 
    # defualtdict will not throw exception when the src not in ticket, will return None instead
    while src in ticket and ticket[src]:
        cur = ticket[src].popleft()
        self.build(cur, ticket, res)
    res.append(src)
    



# 721 
# union-find
""" first time that using dictionary and string as id to implement union find """ 
class Solution(object):
    def accountsMerge(self, accounts):
        """ use the unique element to represent the parent or id, in this case is email """
        parents = {}
        email_to_name = {}
        # set up email and its root
        for account in accounts:
            name = account[0]
            # for each email in the list, if the email not in parent, assign itself as its parent
            # and assign the name to the email in the email_to_name 
            # do union operation, which is union first to the second argument
            for em in account[1:]:
                if em not in parents:
                    parents[em] = em
                email_to_name[em] = name
                self.union(em, account[1], parents)

        import collections
        # collecting all the emails with same root/parent into one list
        components = collections.defaultdict(list)
        for em in parents.keys():
            # what's in the table, is the direct parent not the root, only when two roots merging, the table will update. 
            r = self.find(em, parents)
            components[r].append(em)

        return [[email_to_name[r]] + sorted(l) for r, l in components.items()]

    def find(self, email, parents):
        while email != parents[email]:
            parents[email] = parents[parents[email]]
            email = parents[email]
        return email

    def union(self, e1, e2, parents):
        parents[self.find(e1, parents)] = parents[self.find(e2, parents)]




# 737 
# union find using dictionary implementation
""" tip: 
when a function is returning boolean, think about what case makes the situation false, this is usually easier to find the edge cases. """
def areSentencesSimilarTwo(self, words1, words2, pairs):
    if len(words1) != len(words2): return False
    
    sim = {}
    for w1, w2 in pairs:
        if w1 not in sim: 
            sim[w1] = w1    
        
        if w2 not in sim:
            sim[w2] = w2
        
        self.union(w1, w2, sim)
    
    for i in range(len(words1)):
        # cannot use sim to check parents of two words because, the words maybe not in the sim.
        if self.find(words1[i], sim) != self.find(words2[i], sim): 
            return False 
    return True 
    
    
def find(self, w, sim):
    # this line cover the edge cases that, when two words not in the sim or one of words not in the sim. 
    if w not in sim: return w 
    while w != sim[w]:
        sim[w] = sim[sim[w]]
        w = sim[w]
    return w


def union(self, w1, w2, sim):
    sim[self.find(w2, sim)] = self.find(w1, sim)



# 743 
""" 
dijkstra algorithm, is a greedy algorithm, which picked the least weights from all the ajacent
edges. 
priority queue is used to implement the greedy solution, which is to pick the least weight of all adjacent edges
because of the pq will always maintain the min weight edge destination node on the root

this algorithm essentially is bfs with pq instead of normal queue. 

python: pq is initialized by [], and call it using heapq's heappop and heappush, after heappop called the heap will maitaining the heap invariant. heappush is used to 
build heap from empty list or into an existing heap.

this algorithm is searching thru all the nodes in the graph and find the shortest path in according to wights to get to each node

  """
def networkDelayTime(self, times, N, K):
    import heapq, collections
    # initilize the pq using [], [0] is the priority, in this case, is the time
    """ 
    tips: 
        we are using the time to decide the priority, thus the pq tuple the time should be in front of the node
        which is (0, K)
        dykistra shortest path is just bfs using priority queue, don't forget to maintain a visited dictionary to store
        visited nodes and its weight 
        heapq.heappop() will automatically turn list into a min heap.
        don't forget to inclue the queue in the function of heapq.heappush(pq, node)
    """
    pq, s, adj = [(0, K)], {}, collections.defaultdict(list)
    # initialize the node and its adjacent edges
    for u, v, w in times:
        adj[u].append((v, w))

    while pq:
        time, nxt = heapq.heappop(pq)
        # if one node has multiple relaxation values in the heap, only the smallest one geting pop off first,
        # and the node is added to the final result s dicitonary. and the rest will not be in side this if condition
        if nxt not in s:
            # s is storing each visited nodes and its final shortest path is determined
            s[nxt] = time
            for v, w in adj[nxt]:
                # relax all the edges leaving nxt 
                heapq.heappush(pq, (time+w, v))

    # above procedure will store min time to get to each node from src in the graph. 
    # thus the max value will be the shortest time needed to get to certain node in the graph
    # if that node is reached, then other nodes should have no problem being reached within that time. 
    # therefore we return the max value
    return max(s.values()) if len(len(s)) == N else -1


# bellman ford algorithm implementation
# dfs solution



# 787 
# problem is asking for shortest path to one node and no cycle in the graph,  thus no need to store all the nodes in s like above
def findCheapestPrice(self, n, flights, src, dst, K):
        import heapq, collections
        pq, adj = [(0, src, -1)], collections.defaultdict(list) 

        for u, v, w in flights:
            adj[u].append((v, w))

        while pq:
            price, node, hops = heapq.heappop(pq)
            if node == dst:
                return price
            # no need to append those hops exceeding the k.
            if hops < K:
                # push all neigbhors that within stops to the pq.
                for v, w in adj[node]:
                    heapq.heappush(pq, (price+w, v, hops+1))
        return -1



# 959 
def regionsBySlashes(self, grid):
#   scale the grid 3 times bigger     
    n = len(grid)
    m = len(grid[0])
    bigger_grid = [['1' for _ in range(3*n)] for _ in range(3*m)]
    for i in range(n):
        for j in range(m):
            if grid[i][j] == '/':
                bigger_grid[i*3][j*3+2] = '0'
                bigger_grid[i*3+1][j*3+1] = '0'
                bigger_grid[i*3+2][j*3] = '0'
            elif grid[i][j] == '\\':
                bigger_grid[i*3][j*3] = '0'
                bigger_grid[i*3+1][j*3+1] = '0'
                bigger_grid[i*3+2][j*3+2] = '0'    
    
    count = 0 
    for i in range(3*n):
        for j in range(3*m):
            if bigger_grid[i][j] == '1':
                self.dfs(bigger_grid, i, j)
                count += 1
    return count
                

def dfs(self, grid, i, j):
    if i < 0 or i >= len(grid) or j < 0 or j >= len(grid[0]) or grid[i][j] != '1': return
    
    grid[i][j] = '#'
    for x, y in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
        self.dfs(grid, x+i, y+j)
    

# 134
""" Greedy"""
def canCompleteCircuit(self, gas: List[int], cost: List[int]) -> int:
    start = avail = 0
    for i in range(len(gas)): 
        avail += gas[i] - cost[i]
        if avail < 0:
            start = i+1
            avail = 0
    return start if sum(gas) >= sum(cost) else -1



""" more String/array/math/bit Manipulation """
# 443
def compress(self, chars: List[str]) -> int:
    anchor = ans_idx = 0 
    for i in range(len(chars)+1):
        if i == len(chars) or chars[i] != chars[anchor]:
            chars[ans_idx] = chars[anchor]
            ans_idx += 1
            if i - anchor > 1: 
                if i - anchor < 10:
                    chars[ans_idx] = str(i-anchor)
                    ans_idx += 1
                else:
                    for s in list(str(i - anchor)):
                        chars[ans_idx] = s
                        ans_idx += 1
            anchor = i 
    return ans_idx


# 415
def addStrings(self, num1: str, num2: str) -> str:
    res = []
    carry = 0
    i = len(num1)-1
    j = len(num2)-1
    while i >= 0 or j >= 0 or carry:
        n1 = n2 = 0
        if i >= 0:
            n1 = ord(num1[i]) - ord('0')
        if j >= 0:
            n2 = ord(num2[j]) - ord('0')
        carry, digit = divmod(n1+n2+carry, 10)
        res.insert(0, str(digit))
        i -= 1
        j -= 1
    return ''.join(res)


# 1041
def isRobotBounded(self, instructions: str) -> bool:
    x, y = 0, 0
    cur = 0 # north
    # coz turn left, can also be represented by turn right 3 times 
    clock_wise = [(0, 1), (1, 0), (0, -1), (-1, 0)] # four directions clock wise
    for i in instructions:
        if i == 'R':
            cur = (cur + 1) % 4 # mod 4 in case cur at index 3
        elif i == 'L':
            cur = (cur + 3) % 4 
        else:
            x += clock_wise[cur][0]
            y += clock_wise[cur][1]
    return (x, y) == (0, 0) or cur > 0



# another way
def isRobotBounded(self, instructions: str) -> bool:
    x, y, dx, dy = 0, 0, 0, 1
    for i in instructions:
        if i == 'R': 
            dx, dy = dy, -dx
        if i == 'L': 
            dx, dy = -dy, dx
        if i == 'G': 
            x, y = x + dx, y + dy
    return (x, y) == (0, 0) or (dx, dy) != (0,1)

# 388





# 268
# xor 
# if n not in num, res = 0  and 0 ^ n => n 
# if n in num, res != 0 n ^ n = 0, the missing is left
def missingNumber(self, nums: List[int]) -> int:
    x = i = 0
    while i < len(nums):
        x ^= i 
        x ^= nums[i]
        i += 1
    return x ^ i



# 204
# mark off all primes multiples.
# p*q = n 
# if p <= q: p*p <= n => p <= n**0.5

def countPrimes(self, n: int) -> int:
    prime = [True] * n
    cnt = 0
    for k in range(2, n):
        if prime[k]:
            cnt += 1
            j = k # everything < k, is mark off before by 2, 3, ..k-1
            while k * j < n:
                prime[k*j] = False
                j += 1
    return cnt      


  
# 14
def longestCommonPrefix(self, strs: List[str]) -> str:
    if not strs: return ''
    strs.sort(key=len)
    longest = ''
    first = strs[0] # optimzied
    for i in range(1, len(strs[0])+1):
        prefix = first[:i]
        for w in strs[1:]:
            if w[:i] != prefix:
                return longest
        longest = prefix
    return longest


           
# 609 
# O(n*x) n is the size of input and x is the average length of each input string.
# content => directories + file.txt
def findDuplicate(self, paths: List[str]) -> List[List[str]]:
    dic = collections.defaultdict(list)
    for p in paths: self.convert(p, dic) 
    return [dirlist for dirlist in dic.values() if len(dirlist) > 1]

def convert(self, path, dic): 
    components = path.split()
    for comp in components[1:]:
        tmp = comp.split('.')
        pwd = components[0] + '/' + tmp[0] + '.txt'
        content = tmp[1][4:-1]
        dic[content].append(pwd)

# similar solution
def findDuplicate(self, paths: List[str]) -> List[List[str]]:
    content_dir = collections.defaultdict(list)
    for path in paths:
        parts = path.split()
        prefix = parts[0]
        for p in parts[1:]:
            file = p.split('(')
            filename = file[0]
            content = file[1][:-1]
            content_dir[content].append(prefix + '/'+ filename)   
    return [l for l in content_dir.values() if len(l) > 1]
        

        
# 9
# only reverse half of the digits
# if odd len palindrome, middle digit is the difference by 10, thus reverse//10 should equal to remaining digits.
# multiple of 10, will not be palindrome except 0
def isPalindrome(self, x: int) -> bool:
    if x < 0 or (x != 0 and x % 10 == 0 ): return False
    reverse = 0
    while x > reverse:
        x, d = divmod(x, 10)
        reverse = reverse*10 + d
    return reverse == x or reverse//10 == x



# 412
# in computer, % is more expensive than other operators.
# use add to find multiple of 3, 5 or 3 and 5
def fizzBuzz(self, n: int) -> List[str]:
    res = []
    fizz, buzz = 0, 0 
    for i in range(1, n+1):
        fizz, buzz = fizz+1, buzz+1
        if fizz == 3 and buzz == 5:
            res.append('FizzBuzz')
            fizz = buzz = 0
        elif fizz == 3:
            res.append('Fizz')
            fizz = 0
        elif buzz == 5: 
            res.append('Buzz')
            buzz = 0
        else:
            res.append(str(i))
    return res



# 7
# python mod: always return a number with same sign as denomonator/分母
# ex: -5 % 3
# -5 // 3 => -1.25 => floor(-1.25) => -2
# -5 % 3 => (-2 * 3 + 2) => 2
# if python is not used, thus the int type will overflow, what you can do is reverse the result forming process, if after reversed process, the number returned is not equal to orginal number
# meaning we have overflow to a random garbage number thus we can return 0 
def reverse(self, x: int) -> int:
    res = 0 
    sign = [1, -1][x<0]
    x = abs(x)
    while x:
        x, d = divmod(x, 10)
        res = res*10 + d
        if sign * res < -2**31 or sign*res > (2**31-1): return 0 
    return sign*res




#38
# read n-1 string: count number of same digits + this digit ...
def countAndSay(self, n: int) -> str:
    i = 1
    s = '1'
    while i != n:
        s = self.cntNSay(s)
        i += 1
    return s


# convert s into count and say
def cntNSay(self, s):
    res = ''
    i = 0 
    while i < len(s):
        j = self.cnt_ahead(s, i)
        res += str(j-i) + s[i]
        i = j
    return res


# return index of first char not equal to s[i]
def cnt_ahead(self, s, start):
    i = start
    while i == start or (i-1 >= start and i < len(s) and s[i] == s[i-1]) : i += 1
    return i

    
# 8
def myAtoi(self, st: str) -> int:
        st = st.strip()
        sign = '+'
        if not st: return 0 
        elif st[0] in ['-', '+'] or st[0].isdigit():
            # if sign first
            if st[0] in ['-', '+']:
                if st[0] == '-': 
                    sign = '-'
                num = self.extractNum(1, st)
                if not num: return 0 
                return self.convert(sign, num)
            elif st[0].isdigit():
                num = self.extractNum(0, st) 
                if not num:return 0
                return self.convert(sign, num)   
        else:
            return 0 
 

def extractNum(self, i, st):
    j = i
    # this will check for case such  '--24'
    while j < len(st) and st[j].isdigit():
        j += 1
    return st[i:j]


def convert(self, sign, num):
    if int(num) > 2147483647 and sign == '+': 
        return 2147483647
    elif int(num) > 2147483648 and sign == '-': 
        return -2147483648
    else:
        return int(num) if sign == '+' else -1*int(num)



# 71
# key: dont worry about '/' first, later join with '/'
# join method will only add given delimeter in between.
def simplifyPath(self, path: str) -> str:
    stack = []
    for p in path.split('/'):
        if p == '..':
            if stack:
                stack.pop()
        elif p and p != '.':
            stack.append(p)
    return '/' + '/'.join(stack) 


# bad version
def simplifyPath(self, path: str) -> str:
    stack = []
    i = 0 
    while i < len(path):
        #skip dup / 
        start = i
        while i < len(path) and path[i] == '/': i += 1
        if i > start:
            start = i-1
        #find the end of the word, before next /
        while i < len(path) and path[i] != '/': i += 1
        cur = path[start:i]
        if cur == '/..':
            if stack: stack.pop()
        elif cur not in ['/', '/.']:
            stack.append(cur)
    return ''.join(stack) if stack else '/'



# 12
def intToRoman(self, num: int) -> str:
    values = [ 1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1 ]
    numerals = [ "M", "CM", "D", "CD", "C", "XC", "L", "XL", "X", "IX", "V", "IV", "I" ]
    res = ""
    for i, v in enumerate(values):
        res += (num//v) * numerals[i]
        num %= v
    return res

# 12
# coin change. greedy
def intToRoman(self, num: int) -> str:
    digits = [(1000, "M"), (900, "CM"), (500, "D"), (400, "CD"), (100, "C"), (90, "XC"),(50, "L"), (40, "XL"), (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")]
    roman = []
    for v, sym in digits:
        if num == 0: break
        cnt, num = divmod(num, v)
        roman.append(sym*cnt)
    return ''.join(roman)


# 13
def romanToInt(self, s: str) -> int:
    dic = {'I': 1,'V': 5,'X': 10,'L': 50,'C': 100,'D': 500,'M': 1000}
    total = 0 
    for i in range(len(s)):
        if i+1 < len(s) and dic[s[i+1]] > dic[s[i]]:
            total -= dic[s[i]]
        else:
            total += dic[s[i]]
    return total
            



# 819 
def mostCommonWord(self, paragraph: str, banned: List[str]) -> str:
    for c in "!?',;.": 
        paragraph = paragraph.replace(c, " ")
    paragraph = paragraph.lower()
    counter = collections.defaultdict(int)
    ban_words = set(banned)
    max_word = ''
    max_cnt = 0
    for w in paragraph.split():
        if w not in ban_words:
            counter[w] += 1
            if max_cnt < counter[w]: 
                max_cnt = counter[w]
                max_word = w
    return max_word
        
        
# 937
# letter logs before digit log
# letter logs sorted by lexi, identifier is tie breaker.
# digit log in original order.
def reorderLogFiles(self, logs: List[str]) -> List[str]:
    digits = []
    letters = []
    for log in logs:
        if log.split(' ')[1].isdigit():
            digits.append(log)
        else:
            letters.append(log)

    letters.sort(key = self.customsort)            #lambda x: x.split(' ')[:1]
    letters.sort(key = self.customsort1)          
    result = letters + digits                                  
    return result


def customsort(self, s):
    return s.split(' ')[0]

def customsort1(self, s):
    return s.split(' ')[1:]



# 31
# array
# we must “increase” the sequence as little as possible, thus we loop backward
# decending order sequence does not have next permutation, thus we loop backward to find the 
# first position that is non-decending order, call anchor, the position after it will be decending order. 
# search the array after anchor point, to find the smallest numebr that is greater then
# anchor element. that number will be the smallest number that will cause the sequnce to increase.
# then, reverse the array after the anchor to be accending order. this will create the next permutation.
# T: O(n), S: O(1)

# in simpler way of explainng: 
# find the first position/breaking point from back of the list that next permutation can be generated, as we know non-increasing sequece cannot generate next permuation
# find the smallest digit in a revese sorted array that is greater than the breaking point and swap them
# make the breaking point forward smallest sequnce, which is ascending order. 

def nextPermutation(self, nums: List[int]) -> None:
    anchor = -1
    #loop backward, find the fisrt increasing position
    for i in range(len(nums)-2, -1, -1):
        if nums[i] < nums[i+1]:
            anchor = i
            break
            
    #loop forward fomr anchor+1 to find the last number greater than anchored element
    if anchor != -1:
        for j in range(len(nums)-1, anchor, -1):
            if nums[j] > nums[anchor]:
                nums[j], nums[anchor] = nums[anchor], nums[j]
                break
    
    self.reverse(anchor+1, nums)
            
            
def reverse(self, start, arr):
    end = len(arr)-1
    while start < end:
        arr[start], arr[end] = arr[end], arr[start]
        start += 1
        end -= 1



# 29 
# math
# Both dividend and divisor will be 32-bit signed integers.
# edge case: if dividend is less than -2147483648, return -2147483648
# if dividend is greater than 2147483647, return 2147483647
# substract divisor in a speed of exponential. 1 divisor, 2 divisors, 4 divisors .. 2^x divisors. 
# T:O(logDividend), S:O(1)

def divide(self, dividend: int, divisor: int) -> int:
    sign = [-1, 1][(dividend < 0) == (divisor < 0)]
    dividend, divisor = abs(dividend), abs(divisor)
    quo = 0
    while dividend >= divisor:
        tmp, i = divisor, 1
        while dividend >= tmp:
            dividend -= tmp
            quo += i
            # 2 times speed of prev speed.
            tmp, i = tmp << 1, i << 1
    return min(max(-2147483648, quo*sign), 2147483647)
            




#283 
# two pointers
# next_zero will stop at positions that contains 0
# i will always moving forward and check when element is not 0
# T: o(n), S:O(1)
def moveZeroes(self, nums: List[int]) -> None:
    next_zero = 0
    for i in range(len(nums)):
        if nums[i] != 0:
            if i != next_zero:
                # next_zero position is 0, so just swap, and 0 will be swap to behind.
                nums[next_zero], nums[i] =  nums[i], nums[next_zero]
            next_zero += 1


# only use assignment no swap
def moveZeroes(self, nums: List[int]) -> None:
    nxt = 0 
    for i in range(len(nums)):
        if nums[i] != 0:
            if i != nxt: 
                nums[nxt] = nums[i]
            nxt += 1
            
    for j in range(nxt, len(nums)):
        nums[j] = 0


# 88
# intuition: there will be some elements of nums1 are in the wrong place, should be move the empty space behind.
# rest of elements are in the right place.
# question: how to only move the wrong ones, and leave the right ones stay put. 
# solution:
# always think about "next available position" for two pointers problem. 
# if we compare one by one from the front, it will require all the elements after shifting right, that is costly.
# if we do it backward, only the ones that in the wrong place need to move.
def merge(self, nums1: List[int], m: int, nums2: List[int], n: int) -> None:
    p1, p2 = m-1, n-1
    p = len(nums1)-1
    while p2 >= 0 and p1 >= 0:
        if nums1[p1] <= nums2[p2]:
            nums1[p] = nums2[p2]
            p2 -= 1
        else:
            nums1[p] = nums1[p1]
            p1 -= 1
        p -= 1
    
    if p1 < 0: nums1[:p+1] = nums2[:p2+1]
        

#67
def addBinary(self, a: str, b: str) -> str:
    carry = 0 
    res = []
    p, q = len(a)-1, len(b)-1
    while p >= 0 or q >= 0 or carry > 0:
        total = carry
        total += int(a[p]) if p >= 0 else 0 
        total += int(b[q]) if q >= 0 else 0 
        carry, rem = divmod(total, 2)
        res.insert(0, str(rem))
        p, q = p-1, q-1
    return ''.join(res)
    



#680
# compare string in between only when mismatch found. 
def validPalindrome(self, s: str) -> bool:
    left, right = 0 , len(s)-1
    while left < right:
        if s[left] != s[right]:
            one, two = s[left+1:right+1], s[left:right]
            return one == one[::-1] or two == two[::-1]
        left, right = left+1, right-1
    return True
        



# 953
# lexical order / dictionary order: 
# compare from left to right if possible
# left side has higher weight when comparing order
# compare chars of  w1, w2 pairwise:
    # if c1 == c2: tie, move to next pair
    # if c1 < c2: in order
    # if c1 > c2: out of order
# if one word is prefix of another word, the shorter length word comes first. 
def isAlienSorted(self, words: List[str], order: str) -> bool:
    dic = {c: i for i, c in enumerate(order)}
    words = [[dic[c] for c in w] for w in words]
    for i in range(1, len(words)):
        if not self.compare(words[i-1], words[i]):
            return False
    return True



# once you find a pair that is violate following condition return bool.
def compare(self, w1, w2):
    for c1, c2 in zip(w1, w2):
        if c1 < c2:
            return True 
        if c1 > c2:
            return False
    return len(w1) < len(w2)
        


# 1128
# lee solution
# trick: 高斯求和公式   n(1+n)/2
def numEquivDominoPairs(self, dominoes: List[List[int]]) -> int:
    dic = {}
    for d in dominoes:
        k = min(d[0], d[1]) * 10 + max(d[0], d[1])
        dic[k] = dic.get(k, 0) + 1
    
    cnt = 0 
    for v in dic.values():
        cnt += ((v-1) * v)//2
    return cnt


    #  my solution 
def numEquivDominoPairs(self, dominoes: List[List[int]]) -> int:
    pairs = 0 
    dic = {}
    for d in dominoes:
        key = tuple(sorted(d))
        if key not in dic:
            dic[key] = 1 
        else:
            dic[key] += 1 
            pairs = pairs + dic[key]-1
    return pairs
                      


# 957
# cycle + hashtable
# fastforward to skip repeating states. 
# if there is K cells, there are at most 2^k states, if N > 2^k, meaning there must be some repeating states. 
def prisonAfterNDays(self, cells: List[int], N: int) -> List[int]:
    seen = {}
    while N > 0:
        t = tuple(cells)
        if t in seen:
            # fastforward
            N %= seen[t] - N
        else:
            seen[t] = N
            
        if N > 0:
            N -= 1
            # next state
            cells = [0] + [int(cells[i-1] == cells[i+1]) for i in range(1, len(cells)-1)] + [0]
    return cells





# greatest common divisor or Euclid algorithm
# a > b 
# EUCLID(a, b)
# 1 if b == 0
# 2    return a
# 3 else:
# 4    return EUCLID(b, a mod b)

# apply Euclid 
# 1071
def gcdOfStrings(self, str1: str, str2: str) -> str:
    def gcd(a, b): 
        return a if b == 0 else gcd(b, a%b)
    d = gcd(len(str1), len(str2))
    return str1[:d] if str1[:d] * (len(str2)//d) == str2 and str2[:d] * (len(str1)//d) == str1 else '' 


#  # xxx, xx  -> xx, x -> x, ''
def gcdOfStrings(self, str1: str, str2: str) -> str:
    if not str1 or not str2: 
        return str1 or str2
    elif len(str1) < len(str2): 
        return self.gcdOfStrings(str2, str1)
    elif str1[:len(str2)] == str2: 
        return self.gcdOfStrings(str1[len(str2):], str2)
    else:
        return ''



# 228
# find continuous subarray with increment of 1
def summaryRanges(self, nums: List[int]) -> List[str]:
    res , i = [], 0
    while i < len(nums):
        a = nums[i]
        while i + 1 < len(nums) and nums[i] + 1 == nums[i+1]: i += 1
        
        if nums[i] != a: 
            res.append(str(a) + '->' + str(nums[i]))
        else:
            res.append(str(a))
        i += 1
    return res



# 950
# deck of cards and put at bottom -> queue data structure
# tip: think the cards as the postions, try to place the sorted card values into the positon by simulation
def deckRevealedIncreasing(self, deck: List[int]) -> List[int]:
    target_order = collections.deque(range(len(deck)))
    deck.sort() # this is revealing order
    res = [0]*len(deck)
    # simulate the steps
    for n in deck:
        res[target_order.popleft()] = n
        if target_order:
            target_order.append(target_order.popleft())
    return res
        

# 280
# the peak is at the odd
# valley is at even 
def wiggleSort(self, nums: List[int]) -> None:
    for i in range(len(nums)):
        if i & 1 == 1:
            if nums[i] < nums[i-1]: 
                nums[i], nums[i-1] = nums[i-1], nums[i]
        elif i != 0 and nums[i] > nums[i-1]:
                nums[i], nums[i-1] = nums[i-1], nums[i]




# 229
# trick: there wil be no more than 2 elements are majority
def majorityElement(self, nums: List[int]) -> List[int]:
    times = len(nums)//3
    cnt_one = cnt_two = 0
    one = two = None 
    for i in range(len(nums)):
        if one == nums[i]: cnt_one += 1 
        elif two == nums[i]: cnt_two += 1 
        elif cnt_one == 0: 
            one, cnt_one = nums[i], 1
        elif cnt_two == 0: 
            two, cnt_two = nums[i], 1
        else: cnt_one, cnt_two = cnt_one-1, cnt_two-1
    return [n for n in [one, two] if nums.count(n) > times]
                

# 442
# trick: same number will be mapped to same index, thus if some number repeated, that mapped index will be visited twice. 
# use negative sign to indicate visited before. 
def findDuplicates(self, nums):
    ans = []
    for i in range(len(nums)):
        idx = abs(nums[i])-1
        if nums[idx] < 0:
            ans.append(idx+1)
        nums[idx] = -1*nums[idx]
    return ans


# 443 
def compress(self, chars: List[str]) -> int:
    i = 0 # next unique char pos
    nxt = 0 # next place for compressed string
    while i < len(chars):
        rep = self.lookAhead(i, chars, chars[i])
        chars[nxt] = chars[i] 
        if rep > 1: 
            for d in str(rep):
                nxt += 1
                chars[nxt] = d 
                # nxt will stop at last compressing digit inside this loop
        nxt += 1
        i += rep
    return nxt
                    
                   
# count continous rep
def lookAhead(self, start, chars, target):
    rep = 0
    for i in range(start, len(chars)):
        if chars[i] != target: break
        rep += 1
    return rep      
    


# 50 
def myPow(self, x, n):
    if n < 0:
        n = -n 
        x = 1/x
    return self.my_pow(x, n)

# tip: x^2n = (x^n)^2
def my_pow(self, x, n):
    if n == 0: return 1.0 
    half = self.my_pow(x, n//2)
    if n % 2 == 0: return half * half
    return half*half*x

# iterative
def myPow(self, x, n):
    if n < 0:
        n = -n 
        x = 1/x
    ans = 1
    prod = x
    while n > 0:
        if n & 1 == 1:
            ans *= prod
        prod = prod * prod 
        n >>= 1 
    return ans 
    

# 54 
# intuitive way
# tip:
# use a seen array to check if current cell was visted or not 
# use two array to mandate the order of clockwise. 
# use mod to change direction to starting direction
# check if next position is good to visit.
def spiralOrder(self, matrix):
    if not matrix: return []
    n, m = len(matrix), len(matrix[0])
    seen =[[False] * m for _ in range(n)]
    ans = []
    # clockwise         
    dr = [0, 1, 0, -1]
    dc = [1, 0, -1, 0]
    #starting point
    r = c = di = 0
    # moving n*m times
    for _ in range(n*m):
        ans.append(matrix[r][c])
        seen[r][c] = True
        nr, nc = r + dr[di], c + dc[di]
        if 0 <= nr < n and 0 <= nc < m and not seen[nr][nc]:
            r, c = nr, nc
        else:
            di = (di+1) % 4 
            r, c = r + dr[di], c + dc[di]
    return ans



# Exiting condition: topBottomPointer <= bottomTopPointer and leftRightPointer <= rightLeftPointer:
# because the state of each pointer is changing inside the block and may not be able to go back to first line condition check, so 
# every time pointer state changes, need to check the states. we donot need to check first and last blocks because they will be checked 
# by the while condition.
def spiralOrder(self, matrix):
    if not matrix: return []
    n, m = len(matrix), len(matrix[0])
    # move col by col
    leftRightPointer = 0 
    rightLeftPointer = m-1
    # move row by row
    topBottomPointer = 0
    bottomTopPointer = n-1
    res = []
    while topBottomPointer <= bottomTopPointer and leftRightPointer <= rightLeftPointer:
        #left to right
        for i in range(leftRightPointer, rightLeftPointer+1):
            res.append(matrix[topBottomPointer][i])
        topBottomPointer += 1
        
        # check right after topBottomPointer is incremented
        if topBottomPointer > bottomTopPointer: break
        # top to bottom
        for i in range(topBottomPointer, bottomTopPointer+1):
            res.append(matrix[i][rightLeftPointer])
        rightLeftPointer -= 1

        #check right after rightLeftPointer is decremented
        if leftRightPointer > rightLeftPointer: break
        # right to left
        for i in range(rightLeftPointer, leftRightPointer-1, -1):
            res.append(matrix[bottomTopPointer][i])
        bottomTopPointer -= 1
        
        # bottom to top
        for i in range(bottomTopPointer, topBottomPointer-1, -1):
            res.append(matrix[i][leftRightPointer])
        leftRightPointer += 1
    return res 
        


# oneline
def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
    return matrix and [*matrix.pop(0)] + self.spiralOrder([*zip(*matrix)][::-1])


# 59
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



# 240
# trick: start from top right conner 
# O(m+n) 
def searchMatrix(self, matrix, target):
    if not matrix or not matrix[0]: return False
    n, m = len(matrix), len(matrix[0])
    r, c = 0, len(matrix[0])-1
    
    while 0 <= r < n and 0 <= c < m:
        if matrix[r][c] > target:
            c -= 1
        elif matrix[r][c] < target:
            r += 1
        else:
            return True
    return False



# 289 
# tip: use signal to represent the states change, thus solve this problem in place.
# -1: live -> dead
# 2: dead -> live
# 0, 1 = dead, live
# tricy: when counting 1s', some 1's maybe  turn into -1, but you need to count that in. 
def gameOfLife(self, board):
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1,-1), (-1, 1), (1, -1)]
    n, m = len(board), len(board[0])
    for i in range(n):
        for j in range(m):
            cnt = 0
            for x, y in directions:
                if 0 <= i+x < n and 0 <=j+y < m and abs(board[x+i][y+j]) == 1: 
                    cnt += 1
            
            if board[i][j] == 1 and (cnt > 3 or cnt < 2):
                board[i][j] = -1
                
            if board[i][j] == 0 and cnt == 3: 
                board[i][j] = 2
    
    for i in range(n):
        for j in range(m): 
            if board[i][j] == -1: board[i][j] = 0
            if board[i][j] == 2: board[i][j] = 1
    return board
    


# 238 
# trick: product except self contains the product of left side of cur and right side of cur using two varible to track the left and right side of cur
def productExceptSelf(self, nums: List[int]) -> List[int]:
    res = [1]*len(nums)
    prev = 1
    for i in range(1, len(nums)):
        prev *= nums[i-1]
        res[i] = prev
        
    rem = 1
    for j in range(len(nums)-1, -1, -1):
        res[j] *= rem
        rem *= nums[j]
    return res
                

#859
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


# 917
def reverseOnlyLetters(self, S):
    i, j = 0, len(S)-1
    s = list(S)
    while i < j:
        if s[i].isalpha() and s[j].isalpha():
            s[i], s[j] = s[j], s[i]
            i += 1
            j -= 1
        elif s[i].isalpha():
            j -= 1
        else:
            i += 1 
    return ''.join(s)


# 125
def isPalindrome(self, s):
    i , j = 0, len(s)-1
    while i < j:
        """ skip all the non alphnumeric at once will speed up the algorithm """
        while i < j and not s[i].isalnum():
            i += 1
        while i < j and not s[j].isalnum():
            j -= 1
        # python will not throw exception when digit string calling upper() or lower(), it just do nothing
        if s[i].upper() != s[j].upper(): 
            return False
        i += 1
        j -= 1

    return True 



# 11
# reasoning:
# area is bound by width and shorter heights of i, j. 
# the widest container is best potential candidate to start
# we are searching for higher water level when the width is shrinking, thus we need to keep tall bar and igore the shorter bar.

# special case: 
# when h[i] == h[j]: because (i+1, j) and (i, j-1) are smaller area than (i, j), thus you move either way will not affect the 
# result. if i+1 or  j-1 bar are taller than i or j, their area will be the same, becaue i or j bar will become the shorter bars which will be used to calculate area. 

def maxArea(self, height):
    max_area = 0
    i, j = 0, len(height)-1
    while i < j:
        d = j - i
        h = min(height[i], height[j])
        max_area = max(max_area, d*h)            
        if height[i] > height[j]:
            j -= 1
        else:
            i += 1
    return max_area



# 455
def findContentChildren(self, g, s):
    g.sort()
    s.sort() 
    cnt , i, j = 0, 0, 0
    while i < len(g) and j < len(s):
        if s[j] >= g[i]: 
            cnt += 1
            i += 1
            j += 1
        else:
            j += 1
    return cnt

           

# 925 
def isLongPressedName(self, name, typed):
    i = 0 
    for j in range(len(typed)):
        if i < len(name) and name[i] == typed[j]:
            i += 1
        # when current two char is different, it could be caused by longpressed, so check if current one is the same as the previous one
        elif j == 0 or typed[j] != typed[j-1]:
            return False
    return i == len(name)



# 20
def isValid(self, s: str) -> bool:
    stack = []
    dic = {'[': ']', '(': ')', '{': '}'}
    for i in range(len(s)):
        if s[i] in dic:
            stack.append(s[i])
        else:
            if not stack or dic[stack.pop()] != s[i]: 
                return False
    return not stack


# 150
def evalRPN(self, tokens: List[str]) -> int:
    nums = []
    for t in tokens:
        if t in ['-', '+', '/', '*']:
            second, first = nums.pop(), nums.pop()
            if t == '-':
                nums.append(first - second)
            if t == '+':
                nums.append(first + second)
            elif t == '*': 
                nums.append(first * second)
            elif t == '/':
                nums.append(int(first / second))
        else:
            nums.append(int(t))
    return nums.pop()
    



# citrix OA:
# a, b are strings that has same length, and check how many edits needed to convert b into a.  
# return number of edits
def numberOfEdits(a, b):
    cnt = 0
    dic = [0] * 26
    for c in a:
        dic[ord(c) - ord('a')] += 1
    
    for ch in b:
        dic[ord(ch) - ord('a')] -= 1
        if dic[ord(ch) - ord('a')] < 0:
            cnt += 1
    return cnt


# 829
# xk + k*(k-1)/2 = N 
# xk = N - k*(k-1)/2
# k*(k-1)/2 < N => k*k < 2*N => k < sqrt(2*N)
def consecutiveNumbersSum(self, N: int) -> int:
    cnt = 1
    k = 2
    while k * k < 2*N:
        if (N - k*(k-1)//2) % k == 0: 
            cnt += 1
        k += 1
    return cnt


# 6
# handle level by level and using a vairbale moving up and down to indicate which level the current char will be placed and the horizontal is just simply 
# append to the end because the spaces are just for visual your result dont have to contain spaces.  
def convert(self, s: str, numRows: int) -> str:
    if numRows == 1 or numRows > len(s): return s
    st = [''] * numRows
    level = step = 0
    for c in s:
        st[level] += c
        if level == 0:
            step = 1
        elif level == numRows - 1:
            step = -1
        level += step
    return ''.join(st)


# 468
def validIPAddress(self, IP: str) -> str:
    if self.checkIPV4(IP): return "IPv4"
    if self.checkIPV6(IP): return "IPv6"
    return "Neither"



def checkIPV4(self, ip):
    tokens = ip.split('.') 
    if len(tokens) != 4: 
        return False
    for t in tokens:
        if not t: return False
        if len(t) > 1 and t[0] == '0': 
            return False
        for c in t:
            if not c.isdigit(): 
                return False
        if int(t) > 255: 
            return False
    return True

            

def checkIPV6(self, ip):
    tokens = ip.split(':')
    if len(tokens) != 8: 
        return False
    
    for t in tokens:
        if len(t) > 4 or len(t) < 1: 
            return False
        for c in t:
            if not c.isdigit() and not c.isalpha():
                return False
            else:
                if c.isalpha():
                    if ord(c.lower()) > ord('f'):
                        return False
    return True
               

# 819
def mostCommonWord(self, paragraph: str, banned: List[str]) -> str:
    punc = (' ', '!', '?', ',', ';', '.', '\'')
    banned = set(banned)
    tokens = paragraph.split()
    i = 0 
    while i < len(paragraph):
        if paragraph[i] in punc:
            paragraph = paragraph[:i] + ' '+  paragraph[i+1: ]
        i += 1
    paragraph.strip()
    tokens = paragraph.split()
    dic = {}
    for t in tokens:
        t = t.lower()
        dic[t] = dic.get(t, 0) + 1
    most_comm = 0
    most_cnt = 0 
    for k, v in dic.items():
        if v > most_cnt and k not in banned:
            most_comm = k 
            most_cnt = v
    return most_comm 


# 249
def groupStrings(self, strings: List[str]) -> List[List[str]]:
    dic = collections.defaultdict(list)
    for st in strings:
        dic[len(st)].append(st)

    res = []
    same_dist = collections.defaultdict(list)
    for same_len in dic.values():
        if same_len:
            if len(same_len) == 1 or len(same_len[0]) == 1: 
                res.append(same_len)
            else:
                for w in same_len:
                    key = []
                    for i in range(1, len(w)):
                        d = (ord(w[i]) - ord(w[i-1]) + 26) % 26
                        key.append(d)
                    key = tuple(key)
                    same_dist[key].append(w) 
    return res + list(same_dist.values())
                    
                    
# 539        
def findMinDifference(self, timePoints: List[str]) -> int:
    num = []
    for t in timePoints:
        tokens = t.split(':')
        num.append(int(tokens[0]) * 60 + int(tokens[1]))
    num.sort()
    
    min_diff = 24*60
    for i in range(0, len(num)): 
        min_diff = min(min_diff, ((num[i]+24*60 - num[i-1])%(24*60)))
    return min_diff


# 68
# key to solve the problem is to realize how to evenly distribute the spaces. here we pace the space one by one and circle around and startinng the spaces again. 
# this way spaces will be placed evenly and left will have more spaces than right slots. 
def fullJustify(self, words: List[str], mw: int) -> List[str]:
    res, row = [], []
    chars = 0
    for w in words:
        if chars + len(w) + len(row) > mw: 
            spaces = mw - chars
            for i in range(spaces): 
                row[i % (len(row)-1 or 1)] += ' '
            res.append(''.join(row))
            row = []
            chars = 0
        chars += len(w)
        row.append(w)
    lastrow = ' '.join(row) + ' ' * (mw - chars - len(row)+1)
    return res + [lastrow]


# 273
# english way of saying large number is based on the say(<20 or <100 or <1000) + [big] + say(remain)
def numberToWords(self, num: int) -> str:
    self.to_19 = 'One Two Three Four Five Six Seven Eight Nine Ten Eleven Twelve Thirteen Fourteen Fifteen Sixteen Seventeen Eighteen Nineteen'.split()
    self.to_90 = 'Twenty Thirty Forty Fifty Sixty Seventy Eighty Ninety'.split()
    self.bigs = {1000000000:'Billion', 1000000:'Million', 1000:'Thousand'}
    return ' '.join(self.say(num)) or 'Zero'


def say(self, num): 
    if num < 20: 
        return self.to_19[num//1-1: num]
    if num < 100: 
        return [self.to_90[num//10-2]] + self.say(num % 10)
    if num < 1000: 
        return [self.to_19[num//100-1]] + ['Hundred'] + self.say(num % 100)
    
    for big, st in self.bigs.items(): 
        if num // big > 0: 
            return self.say(num//big) + [st] + self.say(num % big)
