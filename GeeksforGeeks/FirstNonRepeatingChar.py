#!/bin/bash

import sys

def firstNonRepeating(str1):
    n = len(str1)
    k = n - 1   
    count = [0]*256
    while(k >= 0):
        count[ord(str1[k])] += 1
        if count[ord(str1[k])] == 1:
            c = str1[k]
        k -= 1
    return c
str1 = "himanshu"
print(firstNonRepeating(str1))
