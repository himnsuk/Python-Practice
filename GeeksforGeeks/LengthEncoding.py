#!/bin/bash

import sys

def encodeString(str1):
    n = len(str1)
    i = 0
    j = 1
    str2 = ""
    count = 0
    while( j <= n-1):
        if str1[i] == str1[j]:
            count += 1
        else:
            str2 += str1[i] + str(count + 1)
            count = 0
        i += 1
        j += 1
    str2 += str1[i] + str(count + 1)
    print str2
str1 = "wwwwaaadexxxxxx"
encodeString(str1)
