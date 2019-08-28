def quicksort(arr, start, end):
    if start < end: 
        pivot = partition(arr, start, end)
        quicksort(arr, start, pivot-1)
        quicksort(arr, pivot+1, end)


def partition(arr, start, end):
    cur = arr[end]
    last = start - 1 # the element right before the cur
    for i in range(start, end):
        if arr[i] <= cur:
            last += 1
            arr[last], arr[i] = arr[i], arr[last] 
    arr[last + 1] , arr[end] = arr[end] , arr[last + 1] 
    return last + 1 # this index is sorted


arr = [1,3,2,4,7,5]
quicksort(arr, 0, 5)
print(arr)


#  arr[last + 1] , cur = cur , arr[last + 1]
# this is wrong because you assign element reference of arr[last+1] to cur variable 
# but the arr[end] is the one point to the end element. cur now have a cp of arr[last+1] reference
# 
