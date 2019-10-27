

# Boyer–Moore majority vote algorithm 
# don't know why 

def majorityElement(nums):
    maj = nums[0]
    diff = 0
    for i in range(len(nums)):
        if diff == 0: 
            maj = nums[i] 
            diff = 1
        elif nums[i] == maj: 
            diff += 1
        else: 
            diff -= 1
    return maj


# diff = 0 no maj. 
# 
print(majorityElement([1, 2, 2, 3, 5, 2, 2])) 


 
