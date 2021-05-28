# 32451 
# def bubble_sort_while(arr):
#     n = len(arr)-1
#     while n > 0:
#         cur = 1
#         while cur <= n:
#             if arr[cur] < arr[cur-1]: 
#                 arr[cur], arr[cur-1] = arr[cur-1], arr[cur]
#             cur += 1
#         n -= 1
#     print(arr)

# # using a for loop to indicate the last positions of unsorted arr
# def bubble_sort_for(arr):
#     for i in range(len(arr)-1, -1, -1):
#         for j in range(i+1):
#             if j+1 <= i and arr[j] > arr[j+1]:
#                 arr[j], arr[j+1] = arr[j+1], arr[j]
#     print(arr)

def bubble_sort_clean(arr):
    # for all elements, i is the last position to be compared with previous position
    for i in range(len(arr)-1, 0, -1):
        for j in range(i):
            if arr[j] > arr[j+1]:
                 arr[j], arr[j+1] = arr[j+1], arr[j]
    print(arr)

# a = [3,2,4,5,1]
# bubble_sort_clean(a)


def selection_sort(arr):
    for i in range(len(arr)):
        min_idx = i
        for j in range(i, len(arr)):
            if arr[j] < arr[min_idx]:
                min_idx = j
        arr[i], arr[min_idx] = arr[min_idx], arr[i]
    print(arr)

a = [3, 2, 4, 5, 1]
# selection_sort(a)


# insertion sort key is to find the right position in the sorted arr thru swapping
# def insertion_sort(arr):
#     for i in range(1, len(arr)):
#         j = i
#         while j > 0 and arr[j] < arr[j-1]:
#             arr[j], arr[j-1] = arr[j-1], arr[j]
#             j -= 1  
#     print(arr)

# insertion_sort(a)

# reduce number of swap operations
# find the first element that is less than cur_val and insert at current position
def insertion_sort_clean(arr):
    for i in range(1, len(arr)):
        pos = i
        cur_val = arr[pos]
        # shifing the elements to the right
        while pos > 0 and arr[pos-1] > cur_val: 
            arr[pos] = arr[pos-1]
            pos -= 1 
        arr[pos] = cur_val
    print(a)

# insertion_sort_clean(a)


def merge_sort(arr):
    if len(arr) > 1: 
        mid = len(arr)//2
        left, right = arr[:mid], arr[mid:]

        merge_sort(left)
        merge_sort(right)

        left_idx, right_idx = 0, 0 
        main_idx = 0 

        while left_idx < len(left) and right_idx < len(right):
            if left[left_idx] < right[right_idx]:
                arr[main_idx] = left[left_idx]
                left_idx += 1 
            else:
                arr[main_idx] = right[right_idx]
                right_idx += 1 
            main_idx += 1 
        
        while left_idx < len(left):
            arr[main_idx] = left[left_idx]
            main_idx += 1
            left_idx += 1
        
        while right_idx < len(right):
            arr[main_idx] = right[right_idx]
            main_idx += 1
            right_idx += 1
        
    # print(arr)

# merge_sort(a)

def quick_sort(arr):
    def partition(arr, start, end):
        par = start
        pivot = arr[end]

        for i in range(start, end+1):
            if arr[i] <= pivot:
                arr[par], arr[i] = arr[i], arr[par]
                par += 1 
        return par-1 # the pivot element index 
    
    def quick_sort_helper(arr, start, end):
        if start > end: return 
        p = partition(arr, start, end)
        quick_sort_helper(arr, start, p-1)
        quick_sort_helper(arr, p+1, end)

    quick_sort_helper(arr, 0, len(arr)-1)
    print(arr) 


# quick_sort(a)



