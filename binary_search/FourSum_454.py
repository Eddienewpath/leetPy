class Solution:
    def fourSumCount_bad(self, A: List[int], B: List[int], C: List[int], D: List[int]) -> int:
        """
        find Sum(A,B) and Sum(C, D)
        check if there a pair that sum to 0 and increment the count
        result: TLE
        the input size increated from N to N*N due to list of all possible sum pairs
        M = N*N
        O(M*M) time, O(M) space
        """
        count = 0
        S1 = self.getSum(A, B)
        S2 = self.getSum(C, D)
        for n in S1:
            for k in S2:
                if n + k == 0:
                    count += 1
        return count

    def getSum(self, l1, l2):
        sum_arr = []
        for n in l1:
            for k in l2:
                sum_arr.append(n+k)
        return sum_arr


    def fourSumCount(self, A: List[int], B: List[int], C: List[int], D: List[int]) -> int:  
        hashtable = {}
        for a in A:
            for b in B:
                if a + b in hashtable: 
                    hashtable[a+b] += 1
                else:
                    hashtable[a+b] = 1

        count = 0
        for c in C:
            for d in D:
                if -c-d in hashtable:
                    count += hashtable[-c-d]
        
        return count
