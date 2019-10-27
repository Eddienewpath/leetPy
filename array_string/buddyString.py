# def buddyStrings(A, B):
#     if len(A) != len(B): return False
#     if A == B and len(set(A)) < len(A): return True
#     diff = [(a, b) for a, b in zip(A, B) if a != b]
#     return diff == 2 and diff[0] == diff[1][::-1]



# if A == B, there must be dup in the A, B, so that you have something to swap. 
# if len of A is diff than len of B, no case
# looking for A[i] != B[i]

# conclusion: thinking about what string does not qualify for this criteria. Edge case thinking process. 

def buddyStrings(A, B):
    if len(A) != len(B): return False
    if A == B: return len(A) != len(set(B))
    idx , count = [], 0
    # same len and different strings
    for i in range(len(A)):
        if A[i] != B[i]:
            count += 1
            idx.append(i)
        if count > 2: 
            return False
    if count == 1: return False
    return A[idx[0]] == B[idx[1]] and A[idx[1]] == B[idx[0]]



print(buddyStrings('ab', 'ab'))


# edge case: if count is 0, meaning two strings are the same. because the problem mandate to swap once, so in this case
# if two strings are the same and there is not dup in the string, meaning they will not be the same after swap 
# for example [ab, ab] false. but [aab, aab] true
# if count is 1, meaning there is only one place that is different, meaning one string will not be able to swap 
#  if count greater than 2 then, it requires more than 1 swap 