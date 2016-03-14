#!/bin/python

import sys


t = int(raw_input().strip())
for a0 in xrange(t):
    n = int(raw_input().strip())
    p = n
    count = 0
    remainder = 0
    while(p):
      remainder = p%10
      p = p/10
      if remainder > 0 and n%remainder == 0 :
        count += 1
    print count
