import sys

n = int(raw_input().strip())
arr = map(int, raw_input().strip().split(' '))
print arr
res = 0
for a in arr:
    res = res + a

print res
