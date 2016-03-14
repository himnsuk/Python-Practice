#!/bin/python

import sys


t = int(raw_input().strip())
for a0 in xrange(t):
    n = int(raw_input().strip())
    if n >= 0:
      height = 1
      if n%2 == 0:
        k = n/2
        while(k):
          height = height*2 + 1
          k -= 1
        print height
      else:
        p = n/2
        while(p):
          height = height*2 + 1
          p -= 1
        height *= 2
        print height


