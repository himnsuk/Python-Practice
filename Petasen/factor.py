import sys

n = int(raw_input().strip())

arr = []

for i in range(n):
    arr.append(raw_input().strip())

def factor(n):
    return set(reduce(list.__add__,([i, n/i] for i in range (1, int(n**0.5) + 1) if n % i == 0)))


for i in arr:
    print len(factor(i)) - 1
