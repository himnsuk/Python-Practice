#!/bin/bash

import sys

def uniqueChar(str):
    dict = {}
    for i in str:
        if i in dict:
            return False
        else:
            dict[i] = 1

    return True

str = "himansu"
print (uniqueChar(str))

