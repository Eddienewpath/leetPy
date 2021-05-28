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



# quick sort basic
def quick_sort(arr):
    quicksort(0, len(arr)-1, arr)


def quicksort(i, j, arr):
    if i > j: return 
    p = partition(i, j, arr)
    # print(arr)
    quicksort(i, p-1, arr)
    quicksort(p+1, j, arr)

# algorithm book implemetation
# last element as pivot
def partition(left, right, arr):
    par = left # every element before par are < than pivot
    i = left
    pivot = arr[right]
    while i <= right:
        if arr[i] <= pivot:
            arr[i], arr[par] = arr[par], arr[i]
            par += 1
        i += 1
    return par-1


# Hoare partition algorithm: the original version
# first element as pivot
# i: find element > pivot
# j: find element < pivot
# swap(i, j) until i > j, 
# j is the partition point



arr = [9,2,3,45,6,12,1]

# print(merge_sort(arr))

quick_sort(arr)
print(arr)




