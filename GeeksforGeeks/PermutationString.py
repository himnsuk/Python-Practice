#!/bin/bash

import sys

def permutation(arr, l, r):
    if l==r:
        print "".join(arr)
    else:
        for i in xrange(l, r+1):
            a[l], a[i] = a[i], a[l]
            permutation(arr, l+1, r)
            a[l], a[i] = a[i], a[l]

str = "ABC"
r = len(str)
a = list(str)
permutation(a, 0, r-1)
