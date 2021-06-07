# import math
# def calculate_prime_factors(p):
#     factors = []
#     d = 2
#     n = int(math.sqrt(p)) + 1
#     while n > 1:
#         while p % d == 0:
#             factors.append(d)
#             p /= d
#         d = d + 1
#         if d*d > n:
#             if n > 1: factors.append(n)
#             break
#     return max(factors)

# import primefac
#
# import sys
#
# factors = list( primefac.primefac(20) )
# print factors

# def prime_factors(n):
#     factors=[]
#     d=2
#     while(d*d<=n):
#         while(n>1):            
#             while n%d==0:
#                 factors.append(d)
#                 n=n/d
#             d+=1
#     return factors[-1]
# print prime_factors(20)
# print max(prime_factors(20))

# import sys
# import math
#
# def prime_factorize(n):
#     factors = []
#     number = int(math.fabs(n))
#     return number

# def primes_below(n):
#     lis = set([p * i for p in range(2, n + 1) for i in range(2, n + 1)])
#     return sorted(set(range(2, n + 1)) - lis)

from itertools import count

def prime_factors(n):
    factors = []
    for i in count(2):
        while n % i == 0:
            factors.append(i)
            n //= i
        if 2 * i > n:
            return factors

print prime_factors(20)

# def arrange(arr):
#     return ''.join(sorted(arr, cmp=lambda x, y: cmp(y + x, x + y)))
#
# arr = [5, 4, 3, 2]

# print arrange(arr)
