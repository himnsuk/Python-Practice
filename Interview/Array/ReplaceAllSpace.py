#!/bin/bash

import sys

def replaceSpace(str):
    str2 = list(str)
    n = len(str2)
    for i in range(n):
        if str2[i] == " ":
            str2[i] = "%20"

    return ''.join(str2)

str = "himanshu Kesarvani"
print (replaceSpace(str))
