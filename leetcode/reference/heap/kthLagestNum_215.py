class Solution:
    # check out the python3 heap implemntation souce code for detail
    # def findKthLargest(self, nums: List[int], k: int) -> int:
    #     heap = []
    #     for num in nums:
    #         heapq.heappush(heap, num)
    #     for _ in range(len(nums)-k):
    #         heapq.heappop(heap)
    #     return heapq.heappop(heap)

    def findKthLargest(self, nums, k):
        """ implement merge sort """
        self.merge_sort(nums)
        return nums[-k]


    def merge_sort(self, nums):
        arr = [None]*len(nums)
        self.merge_sort_helper(nums, 0, len(nums)-1, arr)


    def merge_sort_helper(self, nums, left, right, arr):
        if left < right:
            mid = (left + right)//2
            self.merge_sort_helper(nums, left, mid, arr)
            self.merge_sort_helper(nums, mid+1, right, arr)
            self.merge(nums, arr, left, mid, right)

    def merge(self, nums, temp, left, mid, right):
        left_start = left
        left_end = mid
        right_start = mid+1
        right_end = right
        idx = 0
        while left_start <= left_end and right_start <= right_end:
            print('infinit...')
            if nums[left_start] and nums[right_start]:
                if nums[left_start] >= nums[right_start]:
                    temp[idx] = nums[right_start]
                    right_start += 1
                    idx += 1
                else:
                    temp[idx] = nums[left_start]
                    left_start += 1
                    idx += 1
        # print(temp)
        if left_start <= left_end:
             for i in range(left_start,left_end+1):
                 temp[idx] = nums[i]
        elif right_start <= right_end:
            for i in range(right_start, right_end+1):
                 temp[idx] = nums[i]
        
        print(temp)
        for i, v in enumerate(temp):
            # print(temp)
            nums[i] = v
        # print(nums)


print(Solution().findKthLargest([3, 2, 1, 5, 6, 4], 2))
