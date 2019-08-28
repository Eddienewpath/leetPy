class Max_Heap:
    def __init__(self, arr):
        self.arr = arr
        self.heap_size = len(arr)  # how many elem in heap are stored in arr

    # float down
    def heapify(self, i):
        if i >= self.heap_size:
            return

        left = 2*i + 1
        right = 2*i + 2
        largest = -1

        if left < self.heap_size and self.arr[left] > self.arr[i]:
            largest = left
        else:
            largest = i

        if right < self.heap_size and self.arr[right] > self.arr[largest]:
            largest = right

        if largest != i:
            self.arr[i], self.arr[largest] = self.arr[largest], self.arr[i]
            self.heapify(largest)

    # [(n/2)+1 ... n] are all leaves of the heap tree
    def build_max_heap(self):
        for i in range(len(self.arr)//2)[::-1]:
            self.heapify(i)

    def sort(self):
        self.build_max_heap()
        n = len(self.arr)
        for i in range(1, n)[::-1]:
            arr[0], arr[i] = arr[i], arr[0]
            self.heap_size -= 1
            self.heapify(0)
