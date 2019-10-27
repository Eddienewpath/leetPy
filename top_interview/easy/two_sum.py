# for [i] in nums, find target - [i] in range[i+1, end]
# def twoSum(nums, target):
#     dic = {}
#     for i in range(len(nums)):
#         find = target - nums[i]
#         if find in dic: 
#             return [i, dic.get(find)]
#         else: 
#             dic[nums[i]] = i
#     return []

def twoSum(nums, target):
    dic = {}
    for i, v in enumerate(nums):
        find = target - v
        if find in dic: 
            return [dic[find], i]
        else:
            dic[v] = i


print(twoSum([3,2,4], 6))
            


