""" two pointers """
# 855, 42

# 11
def maxArea(self, height):
    max_area = 0
    i, j = 0, len(height)-1
    while i < j:
        d = j - i
        h = min(height[i], height[j])
        max_area = max(max_area, d*h)
        if height[i] > height[j]:
            j -= 1
        else:
            i += 1
    return max_area


# 125
def isPalindrome(self, s):
    i , j = 0, len(s)-1
    while i < j:
        """ skip all the non alphnumeric at once will speed up the algorithm """
        while i < j and not s[i].isalnum():
            i += 1
        while i < j and not s[j].isalnum():
            j -= 1
        # python will not throw exception when digit string calling upper() or lower(), it just do nothing
        if s[i].upper() != s[j].upper(): 
            return False
        i += 1
        j -= 1

    return True 
    

# 455
def findContentChildren(self, g, s):
    g.sort()
    s.sort() 
    cnt , i, j = 0, 0, 0
    while i < len(g) and j < len(s):
        if s[j] >= g[i]: 
            cnt += 1
            i += 1
            j += 1
        else:
            j += 1
    return cnt



# 917
def reverseOnlyLetters(self, S):
    i, j = 0, len(S)-1
    s = list(S)
    while i < j:
        if s[i].isalpha() and s[j].isalpha():
            s[i], s[j] = s[j], s[i]
            i += 1
            j -= 1
        elif s[i].isalpha():
            j -= 1
        else:
            i += 1 
    return ''.join(s)
                

# 925 
def isLongPressedName(self, name, typed):
    i = 0 
    for j in range(len(typed)):
        if i < len(name) and name[i] == typed[j]:
            i += 1
        # when current two char is different, it could be caused by longpressed, so check if current one is the same as the previous one
        elif j == 0 or typed[j] != typed[j-1]:
            return False
    return i == len(name)



# 986
def intervalIntersection(self, A, B):
    i, j = 0 , 0 
    res = []
    while i < len(A) and j < len(B): 
        if A[i][0] > B[j][1]:
            j += 1
        elif A[i][1] < B[j][0]:
            i += 1
        else:
            p = self.intersection(A[i], B[j])
            res.append(p)
            if A[i][1] >= B[j][1]: 
                j += 1
            else:
                i += 1           
    return res
               
            
def intersection(self, p1, p2):
    pair = [None, None]
    if p1[0] <= p2[0]:
        pair[0] = p2[0]
    else:
        pair[0] = p1[0]
    
    if p1[1] <= p2[1]:
        pair[1] = p1[1]
    else:
        pair[1] = p2[1]
    return pair


# 167
def twoSum(self, numbers, target):
    i, j = 0, len(numbers)-1
    pair = [-1, -1]
    while i < j: 
        if numbers[i] + numbers[j] == target: 
            return [i + 1, j + 1]
        elif numbers[i] + numbers[j] > target: 
            j -= 1
        else:
            i += 1
    return pair
        
            


# 15
def threeSum(self, nums):
    nums.sort()
    res = []
    n = len(nums) 
    for i in range(n):
        if i-1 >= 0 and nums[i] == nums[i-1]: continue
        j, k = i+1, n - 1
        target = 0 - nums[i]
        while j < k:
            if nums[j] + nums[k] == target:
                # [-2,0,0,2,2] edge case
                while j+1 < k and nums[j] == nums[j+1]: j += 1
                while k-1 > j and nums[k] == nums[k-1]: k -= 1
                res.append([nums[i], nums[j], nums[k]])
                j += 1
                k -= 1
            elif nums[j] + nums[k] < target:
                j += 1
            else:
                k -= 1
    return res 
                
