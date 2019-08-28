class Solution:
    def findDuplicate(self, nums: List[int]) -> int:
        """
        n+1 number in [1-n]
        there is only one element being duplicated many times
        cannot modify the given array
        runtime better than O(n*n)
        use O(1) space 

        ex: 
        [1,2,2,3,4] in the range of 1-4
        brute force way create a hashtable see which item has freq > 1 
        O(n) space and O(n) time

        arr[arr[0]] = 2
        arr[arr[1]] = 2
        arr[arr[2]] = 2
        arr[arr[3]] = 3
        arr[arr[4]] = 4
        """

