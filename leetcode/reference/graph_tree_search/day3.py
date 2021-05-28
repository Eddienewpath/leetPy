
# 127
import collections
import string
""" bfs solution, replace every char in beginword with other 25 chars and check if replaced word is in dictionary
update length of the path at each word and remove the word if in the dict to avoid repeating path """
def ladderLength(self, beginWord, endWord, wordList):
    dic = set(wordList)
    queue = collections.deque()
    queue.append([beginWord, 1])
    while queue:
        w, length = queue.popleft()
        for i in range(len(w)):
            for c in string.ascii_lowercase:
                tmp = w[:i] + c + w[i+1:]
                if tmp not in dic:
                    continue
                if tmp == endWord:
                    return length+1
                queue.append([tmp, length+1])
                dic.remove(tmp)
    return 0

""" easy to understand version, bfs template """

def ladderLength_template(self, beginWord, endWord, wordList):
    word_dic = set(wordList)
    if endWord not in word_dic:
        return 0
    queue = collections.deque()
    queue.append(beginWord)
    steps, n = 0, len(beginWord)
    while queue:
        size = len(queue)
        while size:
            front = queue.popleft()
            if front == endWord:
                return steps+1
            for i in range(n):
                for c in string.ascii_lowercase:
                    if c == front[i]:
                        continue
                    cur = front[:i] + c + front[i+1:]
                    if cur in word_dic:
                        word_dic.remove(cur)
                        queue.append(cur)
            size -= 1
        steps += 1
    return 0



""" two-end bfs """
def ladderLength(self, beginWord, endWord, wordList):
    words = set(wordList)
    front, back = {beginWord}, {endWord}
    level = 1 
    while front:
        tmp = set()
        level += 1
        for w in front:
            for i in range(len(w)):
                for c in string.ascii_lowercase:
                    if c != w[i]: 
                        cur = w[:i] + c + w[i+1:]
                        # meaning two end bfs meet at same word, then current level is the result
                        if cur in back:
                            return level
                        if cur in words and cur != endWord:
                            tmp.add(cur)
                            words.remove(cur)
        front = tmp 
        # always pick the smaller group to process
        if len(front) >  len(back):
            front, back = back, front
    return 0 



# 241
# divide and conquar 
# key is recursively calcuate the cartitian products of two partitions
def diffWaysToCompute(self, input):
        if input.isdigit():
            return [int(input)]

        res = [] # be careful not to put this inside the loop 
        # iterate thru the input, and partition the input into two parts for every operator. 
        for i in range(len(input)):
            if input[i] in '+-*':
                # divide and conquar
                res1 = self.diffWaysToCompute(input[:i])
                res2 = self.diffWaysToCompute(input[i+1:])

                for n in res1:
                    for m in res2:
                        res.append(self.process(n, m, input[i]))
        return res

def process(self, left, right, op):
        if op == '+':
            return left + right
        elif op == '-':
            return left - right
        else:
            return left * right



""" DP solution """






# 842
def splitIntoFibonacci(self, S):
        n = len(S)
        # try all the length of first two numbers. 
        # according the description, each number is at most 10 digits long, size of the integer represetation
        for i in range(1, 11):
            for j in range(1, 11):
                # if first two num length is equal or longer than the length of the input, then not able to form fib, so we try another pairs
                if i + j >= n: continue
                res =  self.build_fib(i, j, S)
                # res could be empty list
                if len(res) >= 3: 
                    return res
        return []
                
                
            
def build_fib(self, i, j, s):
    a, b = s[:i], s[i:i+j]
    # check if a number is start with 0 
    if a[0] == '0' and i > 1: return []
    if b[0] == '0' and j > 1: return []
    
    n = len(s)
    first, second = int(a), int(b)
    arr = [first, second]
    offset = i + j
    while offset < n:
        tmp = first + second
        third = str(tmp)
        k = len(third)

        if  third != s[offset : offset+k]: 
            return []
        third = int(third)
        if third > pow(2, 31)-1: 
            return []
        arr.append(third)
        offset += k
        first, second = second, third
    return arr
        
        
""" backtracking solution """


def splitIntoFibonacci_dfs(self, S):
    res = []
    self.can_split(S, 0, res)
    return res


def can_split(self, s, start, path): 
    if start == len(s) and len(path) >=3: 
        return True
    
    for i in range(start, len(s)):
        if i > start and s[start] == '0': 
            break
        
        num = int(s[start : i+1])
        
        if num > 2 ** 31 - 1: 
            break
            
        size = len(path)
        if size >= 2 and num > path[size-1] + path[size-2]:
            break
        # here we cannot code it like this ' path + [num] ' becaue this will create a new list object assgin to the paremeter res. 
        # thus this res is not pointing to the same object as the calling function res.  
        if size < 2 or num == path[size-1] + path[size-2]:
            path.append(num)
            if self.can_split(s, i+1, path):
                return True 
            path.pop()

    return False 
