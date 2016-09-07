#!/bin/bash

import sys

def atoi(str):
    res = 0
    for i in str:
        if ord(i) > 47 and ord(i) < 58:
            res = res*10 + (ord(i) - ord('0'))
    return res

str = "324164526"
print atoi(str)

