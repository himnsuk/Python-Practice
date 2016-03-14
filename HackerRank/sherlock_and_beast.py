#!/bin/python

import sys


t = int(raw_input().strip())
for a0 in xrange(t):
    n = int(raw_input().strip())
    j = n
    result1 = ""
    result2 = ""
    while(j):
        if j%3 == 0 and (n-j)%5 == 0:
            result1 = (j) * "5" + (n - j) * "3"
            break
        j -= 1
    if result1 == "":
        result1 = 0
    else:
        result1 = int(result1)
    j = n
    while(j):
        if j%5 == 0 and (n-j)%3 == 0:
            result2 = (n - j) * "5" + j * "3"
            break
        j -= 1
    if result2 == "":
        result2 = 0
    else:
        result2 = int(result2)
    if result1 == 0 and result2 == 0:
        print -1
    elif result1 >= result2:
        print result1
    else:
        print result2



