# '23:59' set [2,3,5,9]
# 
# def nextClosestTime(time):
# 	words = sorted(list(set(time[:2] + time[3:])))
# 	res = list(time[:2] + time[3:])
#     # idx [3,2,1,0] => word[2,3,5,9] res = [2,3,5,9]
# 	for idx in range(4)[::-1]:
# 		next_idx = words.index(res[idx]) + 1 
# 		if next_idx < len(words):
# 			# change to the next digit
# 			res[idx] = words[next_idx]
# 			hour, minute = divmod(int(''.join(res)), 100)
# 			# if it is valid time, just return it
# 			if hour < 24 and minute < 60:
# 				return ''.join(res[:2]) + ':' + ''.join(res[2:])
# 		# back to the first digit, and carry to high-position
# 		res[idx] = words[0] #[2,2,2,2]
# 	return ''.join(res[:2]) + ':' + ''.join(res[2:])


def nextClosestTime(time):
    ori = set(time) # included the : 
    hr = int(time[:2])
    mi = int(time[3:])
    while True:
        mi += 1
        if mi > 59: 
            mi = 0
            hr = 0 if hr == 23 else hr+1
        if hr < 10 and mi < 10: 
            t = f'0{hr}:0{mi}'
        elif hr < 10:
            t = f'0{hr}:{mi}'
        elif mi < 10:
            t = f'{hr}:0{mi}'
        else:
            t = f'{hr}:{mi}'
        if ori >= set(t):
            return t
    return time
        


print(nextClosestTime('23:59'))
