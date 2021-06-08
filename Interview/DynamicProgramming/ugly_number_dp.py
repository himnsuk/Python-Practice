# Ugly Numbers
# Ugly numbers are numbers whose only prime factors are 2, 3 or 5. The sequence 1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, … shows the first 11 ugly numbers. By convention, 1 is included. 
# Given a number n, the task is to find n’th Ugly number.

# Brute-Force Method
# The time complexity is between O(NlongN) and O(N^2)
# def divide_number(a, b):
#     while a % b == 0:
#         a = a/b
#     return a

# def is_ugly(n):
#     n = divide_number(n, 2)
#     n = divide_number(n, 3)
#     n = divide_number(n, 5)
#     return 1 if n == 1 else 0

# def getNthUglyNumber(n):
#     i = 1

#     count = 1

#     while count < n:
#         i += 1
#         if is_ugly(i):
#             count += 1
#     return i

# print(getNthUglyNumber(11))
# print(getNthUglyNumber(150))

# Dynamic Programming way ugly number

def find_ugly_number(n):
    ugly = [0]*n
    ugly[0] = 1
    i2 = i3 = i5 = 0

    next_multiple_of_2 = ugly[i2] * 2
    next_multiple_of_3 = ugly[i3] * 3
    next_multiple_of_5 = ugly[i5] * 5

    for i in range(1, n):
        next_ugly_number = min(next_multiple_of_2, next_multiple_of_3, next_multiple_of_5)
        ugly[i] = next_ugly_number
        if(next_ugly_number == next_multiple_of_2):
            i2 = i2 + 1
            next_multiple_of_2 = ugly[i2] * 2

        if(next_ugly_number == next_multiple_of_3):
            i3 = i3 + 1
            next_multiple_of_3 = ugly[i3] * 3

        if(next_ugly_number == next_multiple_of_5):
            i5 = i5 + 1
            next_multiple_of_5 = ugly[i5] * 5
        
    return next_ugly_number

print(find_ugly_number(11))