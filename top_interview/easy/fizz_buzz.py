# def fizzBuzz(n):
#     res = []
#     for i in range(1, n+1):
#         if i % 15 == 0: 
#             res.append('FizzBuzz')
#         elif i % 3 == 0:
#             res.append('Fizz')
#         elif i % 5 == 0:
#             res.append('Buzz')
#         else:
#             res.append(str(i))
#     return res 


def fizzBuzz(n):
    # if i%3 == 0, 0 is value to falsey, not 0, is truthey or evaluate to 1. 
    # if divisible by 3, fizz + '', if divisible by 5, '' + buzz, divisible by both, fizz + buzz, else is i 
    return ['Fizz'*(not i%3) + 'Buzz'*(not i%5) or str(i) for i in range(1, n+1)]


print(fizzBuzz(15))
