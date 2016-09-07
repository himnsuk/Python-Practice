#!/bin/bash

import sys

def checkAnagaram(str1, str2):
    count = [0]*256
    n1 = len(str1)
    n2 = len(str2)
    if n1 != n2:
        return False
    else:
        for i in str1:
            count[ord(i)] = 1
        for j in str2:
            if count[ord(j)] != 1:
                return False

        return True

str1 = "abcde"
str2 = "bacde"
print checkAnagaram(str1, str2)
