# in sorted array return index of the target if exists eles return -1
def binary_search_I(arr, target):
    left, right = 0, len(arr)-1
    while left <= right:
        mid = (left + right) >> 1
        if arr[mid] == target: 
            return mid
        elif arr[mid] > target:
            right = mid - 1
        else:
            left = mid + 1
    return -1


# [l, r)
def binary_search_II(arr, target):
    left, right = 0, len(arr)
    while left < right:
        mid = (left + right) >> 1
        if arr[mid] < target:
            left = mid + 1 
        else:
            right = mid
    return left if 0 <= left < len(arr) and arr[left] == target else -1



def binary_search_III(arr, target):
    pass 


# if duplicates allowed in a sorted array, find the left most of the dup of target
def find_lefemost(arr, target):
    pass 


# find the rightmost dup target in the sorted array
def find_rightmost(arr, target):
    pass 



arr = [1,2,3,4,5,6,7,8]
t = 0
print(binary_search_II(arr, t))
