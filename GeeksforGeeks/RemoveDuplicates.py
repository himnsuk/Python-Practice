#!/bin/bash

import sys

p = 256

def removeDuplicate(str):
    count = [0]*p

    for i in str:
        count[ord(i)] += 1

    str2 = ''
    for i in str:
        if count[ord(i)] > 0:
            str2 += i
            count[ord(i)] = 0
    return str2

str = "geeksforgeeks"
print removeDuplicate(str)
