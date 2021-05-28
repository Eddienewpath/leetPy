""" 
three templates
"""
# 1
# [l, r]
def binary_search_i(nums, target):
    if not nums: return -1
    left, right = 0, len(nums)-1

    while left <= right:
        mid = (left + right) // 2 
        if nums[mid] == target:
            return mid
        elif nums[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    return -1 



# 2
# [l,r]
# ending condition left + 1 == right
# two elements left and we just need to check both of them.
def binary_search_iii(nums, target):
    if not nums:
        return -1
    left, right = 0, len(nums)-1

    while left + 1 < right:
        mid = (left + right) // 2
        if nums[mid] < target:
            left = mid
        else:
            right = mid

    if nums[left] == target:
        return left
    if nums[right] == target:
        return right
    return -1



# 3
# [l, r)
def binary_search_ii(nums, target):
    if not nums: return -1
    left, right = 0, len(nums)
    
    while left < right:
        mid = (left + right) // 2 
        if nums[mid] < target:
            left = mid + 1
        else:
            right = mid

    if nums[left] == target: return left
    return -1



""" 
sorted array contains duplicates 
1. find the first number that is >= target
2. find the first number that is > target
"""

# lower bound
def binary_search_lower_bound(nums, target):
    if not nums: return -1
    left, right = 0, len(nums)
    
    while left < right:
        mid = (left + right) // 2
        if nums[mid] < target:
            left = mid + 1
        else:
            right = mid
    
    if nums[left] == target:
        return left
    return -1

# higher bound
def binary_search_higher_bound(nums, target):
    if not nums: return -1
    left, right = 0, len(nums)
    
    while left < right:
        mid = (left + right) // 2
        if nums[mid] <= target:
            left = mid + 1
        else:
            right = mid
    
    if nums[left-1] == target:
        return left-1
    return -1

nums = [1, 2, 3, 3, 3, 4, 5, 6, 7]
target = 3

# print(binary_search_iii(nums, target))  # 5

print(binary_search_lower_bound(nums, target)) # 2 
print(binary_search_higher_bound(nums, target)) # 4
