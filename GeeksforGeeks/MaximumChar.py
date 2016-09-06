#!/bin/bash

import sys

p = 256
def getMaxCharCount(str):
    count = [0]*p
    for i in str:
        count[ord(i)] += 1

    max = 1
    for i in str:
        if max < count[ord(i)]:
            max = count[ord(i)]
            c = i

    return c

str = "GeeksForGeeks"
print getMaxCharCount(str)

