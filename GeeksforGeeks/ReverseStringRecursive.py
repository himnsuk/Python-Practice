#!/bin/bash

import sys

def toMutable(str):
    temp = list(str)
    return temp

def recursiveReverse(str, left, right):
    if(left < right):
        str[left], str[right] = str[right], str[left]
        left += 1
        right -= 1
        recursiveReverse(mstr, left, right)

str = raw_input("Enter the string you want to reverse")
n = len(str)
mstr = toMutable(str)
recursiveReverse(mstr, 0, n - 1)
print str
print ''.join(mstr)
