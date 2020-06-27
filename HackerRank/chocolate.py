import sys


t = int(raw_input().strip())
for a0 in xrange(t):
    n,c,m = raw_input().strip().split(' ')
    n,c,m = [int(n),int(c),int(m)]
    chocolate = n/c
    wrapper = chocolate
    while wrapper >= m:
      chocolate += wrapper/m
      wrapper = wrapper/m + wrapper%m

    print(chocolate)