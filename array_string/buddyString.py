def buddyStrings(A, B):
    if len(A) != len(B): return False
    if A == B and len(set(A)) < len(A): return True
    diff = [(a, b) for a, b in zip(A, B) if a != b]
    return diff == 2 and diff[0] == diff[1][::-1]



# if A == B, there must be dup in the A, B, so that you have something to swap. 
# if len of A is diff than len of B, no case
# looking for A[i] != B[i]


print(buddyStrings('aab', 'aba'))