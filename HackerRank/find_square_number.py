import math
import sys

n = int(raw_input().strip())
for r in xrange(n):
  arr = map(int, raw_input().strip().split(' '))
  count = 0
  for i in range(arr[0], arr[1] + 1):
    x = int(math.sqrt(i)) 
    if i >= 0 and x**2 == i:
      count += 1
  print count
