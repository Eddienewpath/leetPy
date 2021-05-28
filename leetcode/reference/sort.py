# p148
class ListNode(object):
    def __init__(self, x):
        self.val = x
        self.next = None

def sortList(head):
    def find_mid(h):
        slow, fast, prev = h, h, None
        while fast and fast.next:
            prev, slow, fast = slow, slow.next, fast.next.next
        return (prev, slow)

    def merge_sort(head):
        if not head or not head.next: 
            return head
        prev, mid = find_mid(head)
        prev.next = None
        left,right= head, mid
        left = merge_sort(left)
        right = merge_sort(right)
        return merge(left, right)

    def merge(left, right):
        ans = ListNode(-1)
        dummy = ans
        while left and right:
            if left.val < right.val:
                dummy.next = left
                left = left.next
            else:
                dummy.next = right
                right = right.next
            dummy = dummy.next

        dummy.next = left or right
        return ans.next

    return merge_sort(head)



# p56 
# key: 
# sort the array by the first number in the array. python sort() will do that.
# check if next interval has overlap with previous one, if does, update the second element, else add the new interval to the result
# and update the current interval.
def merge(intervals):
    ans = []
    if intervals: 
        intervals.sort()
        cur = intervals[0] 
        ans.append(cur)
        for interval in intervals:
            if interval[0] <= cur[1]:
                cur[1] = max(cur[1], interval[1])
            else:
                cur = interval
                ans.append(interval)
    return ans



# p147
class ListNode(object):
    def __init__(self, x):
        self.val = x
        self.next = None

# using pre and pre.next to abstract the insertion point
def insertionSortList(head):
    if not head: return head
    cur, dummy, nxt = head, ListNode(-1), None
    pre = dummy
    while cur: 
        nxt = cur.next
        # find the first node has that value greater than cur val
        while pre.next and pre.next.val < cur.val: 
            pre = pre.next
        
        # insert current node in between pre and pre.next
        cur.next = pre.next
        pre.next = cur
        # update the pre to the begining of the list once one insertion is done 
        pre = dummy
        cur = nxt
    return dummy.next


# p179
# key is to override the comparison function.
def largestNumber(nums):
    if sum(nums) == 0:
        return '0'
    
    def compare(x, y):
        return str(x) + str(y) > str(y) + str(x)

    def selection_sort(nums):
        for i in range(len(nums)):
            min_idx = i
            for j in range(i, len(nums)):
                if compare(nums[j], nums[min_idx]): 
                    min_idx = j
            nums[i], nums[min_idx] = nums[min_idx], nums[i]
        return ''.join(map(str, nums))

    def bubble_sort(nums):
        for i in range(len(nums)-1, 0, -1):
            for j in range(i):
                if not compare(nums[j], nums[j+1]): 
                    nums[j], nums[j+1] = nums[j+1], nums[j]
        return ''.join(map(str, nums))

    def insertion_sort(nums):
        for i in range(len(nums)):
            cur_val = nums[i]
            cur_pos = i 
            while cur_pos > 0 and compare(cur_val, nums[cur_pos-1]):
                nums[cur_pos] = nums[cur_pos-1]
                cur_pos -= 1 
            nums[cur_pos] = cur_val
        return ''.join(map(str, nums))

    def merge_sort(nums):
        def merge(left, right):
            res = [None]*(len(left) + len(right))
            i, j, k = 0, 0, 0

            while i < len(left) and j < len(right):
                if compare(left[i], right[j]):
                    res[k] = left[i]
                    i += 1 
                else:
                    res[k] = right[j]
                    j += 1
                k += 1

            while i < len(left):
                res[k] = left[i]
                i += 1 
                k += 1 
            
            while j < len(right):
                res[k] = right[j]
                j += 1 
                k += 1 
            return res
        
        def merge_sort_helper(nums):     
            if len(nums) <= 1: return nums
            mid = len(nums)//2
            left = merge_sort_helper(nums[:mid])
            right = merge_sort_helper(nums[mid:])
            return merge(left, right)

        ans = merge_sort_helper(nums)
        return ''.join(map(str, ans))

    def quick_sort(nums):
        def partition(nums, start, end):
            par = start
            pivot = nums[end]
            for i in range(start, end+1):
                if compare(nums[i], pivot):
                    nums[par], nums[i] = nums[i], nums[par]
                    par += 1
            return par-1 

        def quick_sort_helper(nums, start, end):
            if start >= end: return 
            p = partition(nums, start, end)
            quick_sort_helper(nums, start, p-1)
            quick_sort_helper(nums, p+1, end)
        
        quick_sort_helper(nums)
        return ''.join(map(str, nums))
    
    return selection_sort(nums)

    

