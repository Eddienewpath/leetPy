# naive 
def missingNumber(nums):
    total = 0
    for n in range(len(nums)+1):
        total += n
    return total - sum(nums)


# bit manipulation
# a xor b xor b => a 
# n number from [0, n], missing one number, so the largest index of the array is n-1, so n is good to initialize the res. 
# if missing number is n, then res = n. if missing number is not n, so n will be xor out. 
def missingNumber_bit(nums):
    res = len(nums)
    for i in range(len(nums)):
        res ^= i ^ nums[i]

    return res