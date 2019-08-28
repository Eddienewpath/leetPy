
# find the first target
def find_leftmost_dup(nums, n, t):
    l, r = 0, n-1
    while l < r: 
        m = (l+r)//2
        if nums[m] < t:
            l = m + 1
        else:
            r = m
    return l


# the first number that is greater than target
def find_rightmost_dup(nums, n, t):
    l, r = 0, n-1
    while l < r:
        m = (l+r)//2
        if nums[m] <= t:
            l = m + 1
        else:
            r = m
    return l-1


print(find_leftmost_dup([1,2,3,3,3,4,5], 7, 3))  #2 
print(find_rightmost_dup([1, 2, 3, 3, 3, 4, 5], 7, 3)) #4

