
"""
Think of it as if you're filling up a basket with bricks. The basket's depth is the dividend; each brick's length is the divisor's magnitude.
The % operator in Python yields integers with the same sign as the divisor.
"""

def findMinDifference(timePoints):
    def convertToMin(time):
            return int(time[:2])*60 + int(time[3:])
# apply given function to each elements in the list
    time_list = map(convertToMin, timePoints)
    time_list.sort()
# The % operator in Python yields integers with the same sign as the divisor.
    return min((y - x) % (60 * 24) for x , y in zip(time_list, time_list[1:] + time_list[:1]))

print(findMinDifference(["12:12", "00:13"]))






