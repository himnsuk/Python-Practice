#!/bin/bash

import sys

p = 256

def removeDirtyChar(mask_str, real_str):
    count = [0]*p

    for i in mask_str:
        count[ord(i)] += 1

    str2 = ""
    for i in real_str:
        if count[ord(i)] == 0:
            str2 += i
    return str2

mask_str = "mask"
real_str = "geeksforgeeks"
print removeDirtyChar(mask_str, real_str)
