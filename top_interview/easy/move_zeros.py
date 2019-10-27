# i will move forward only when nums[j] is 0, to find the first non-zero
# if num[j] is not 0, nums[i] is the first non-zero meaning cur number is at its right place
# increment both i and j

def moveZeroes(nums):
    j = 0 # next pos for non-zero
    for i in range(len(nums)):
        if nums[i]: 
            nums[i], nums[j] = nums[j], nums[i]
            j += 1
     


a = [0, 1, 0, 3, 12]
moveZeroes(a)
print(a)   
