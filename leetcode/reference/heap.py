# min heap
class Min_Heap:
    def __init__(self):
        self.heap = []


    def heappush(self, item):
        self.heap.append(item)
        self._siftdown(self.heap, 0, len(self.heap)-1)


    def heappop(self):
        last = self.heap.pop() # if the heap is empty raise an error
        if self.heap:
            min_item = self.heap[0]
            self.heap[0] = last
            self._siftup(self.heap, 0)
            return min_item
        return last


    # transform a list into a heap
    def heapify(self, alist):
        n = len(alist)
        # starts from the last item of the first level above the leaves
        for i in range(n//2-1, -1, -1):
            self._siftup(alist, i)

    

    # pos is the position the heap is out of order
    # this function is used to restore the heap invariant
    def _siftdown(self, heap, start_pos, pos):
        item = heap[pos]
        while pos > start_pos:
            parent_pos = (pos - 1)>>1 #get parent position
            parent_item = heap[parent_pos]
            if parent_item > item:
                heap[pos] = parent_item
                pos = parent_pos
                continue
            break
        heap[pos] = item


    # item at pos the heap variant is not maintained
    # children of pos are alreay heap
    # so we move all the children up until pos hit the leaf
    # this algoritm reduces the total number of comparisons, instead of comparing every children with the item at pos and swap 
    def _siftup(self, heap, pos):
        item = heap[pos]
        start_pos = pos
        end_pos = len(heap)
        child_pos = 2*pos + 1 # left child
        while child_pos < end_pos:
            right_pos = child_pos + 1 
            if right_pos < end_pos and heap[child_pos] >= heap[right_pos]:
                child_pos = right_pos
            heap[pos] = heap[child_pos]
            pos = child_pos
            child_pos = 2*pos + 1
        # pos now is on the leaf
        heap[pos] = item
        self._siftdown(heap, start_pos, pos)
        





mh = Min_Heap()
for i in [3,5,7,2,1]:
    mh.heappush(i)
print(mh.heap)
print("test heappush", mh.heap[0], 1)
print("test heappop", mh.heappop(), 1)
print(mh.heap)
alist = [3, 5, 6, 7 ,0 ,1]
mh.heapify(alist)
print(alist)