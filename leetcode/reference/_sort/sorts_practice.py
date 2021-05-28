def selection_sort(arr):
    for i in range(len(arr)):
        min_idx = i
        for j in range(i+1, len(arr)):
            if arr[j] < arr[min_idx]:
                min_idx = j
        arr[i], arr[min_idx] = arr[min_idx], arr[i]
    print(arr)


def bubble_sort(arr):
    for i in range(len(arr)-1, 0, -1):
        for j in range(i):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    print(arr)


def insertion_sort(arr):
    for i in range(1, len(arr)):
        cur_val = arr[i]
        cur_pos = i
        # every thing greater than cur_val will be shift right one step 
        while cur_pos > 0 and arr[cur_pos-1] > cur_val: 
            arr[cur_pos] = arr[cur_pos-1]
            cur_pos -= 1
        # found the insertion pos and assign cur val to this pos.
        arr[cur_pos] = cur_val
    print(arr)


def merge_sort(arr):
    if len(arr) > 1: 
        mid = len(arr)//2
        left = arr[:mid]
        right = arr[mid:]
        merge_sort(left)
        merge_sort(right)

        i, j = 0, 0
        k = 0
        while i < len(left) and j < len(right):
            if left[i] < right[j]:
                arr[k] = left[i]
                i += 1
            else:
                arr[k] = right[j]
                j +=1
            k += 1
        
        while i < len(left):
            arr[k] = left[i]
            k += 1
            i += 1
        
        while j < len(right):
            arr[k] = right[j]
            k += 1
            j += 1



def quick_sort(arr):
    def partition(arr, start, end):
        par = start
        pivot = arr[end]
        for i in range(start, end+1):
            if arr[i] <= pivot:
                arr[i], arr[par] = arr[par], arr[i]
                par += 1
        return par - 1

    def quick_sort_helper(arr, start, end):
        if start >= end: return
        p = partition(arr, start , end)
        quick_sort_helper(arr, start, p-1)
        quick_sort_helper(arr, p+1, end)

    quick_sort_helper(arr, 0, len(arr)-1)
    print(arr)  

arr = [3,2,5,1,7,4,6]
# selection_sort(arr)
# bubble_sort(arr)
# insertion_sort(arr)
# merge_sort(arr)
# print(arr)
quick_sort(arr)





