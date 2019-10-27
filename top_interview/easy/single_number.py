def singleNumber(nums):
    res = nums[0]
    for i in range(1, len(nums)):
        res ^= nums[i]
    return res

print(singleNumber([2,2,1]))

