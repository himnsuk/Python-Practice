#!/bin/bash

import sys

def reverse(str):
    n = len(str)
    i = 0
    j = n - 1
    while(i < j):
        str[i], str[j] = str[j], str[i]
        i += 1
        j -= 1
    return ''.join(str)

str = list("himansu")
print (reverse(str))
