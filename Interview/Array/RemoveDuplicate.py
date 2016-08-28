#!/bin/bash

import sys

def removeDuplicate(str):
    n = len(str)
    for i in range(n):
        for j in range(i+1, n - 1):
            if str[i] == str[j]:
                del str[j]

    return ''.join(str)

str = list("himanshu")
print (removeDuplicate(str))
