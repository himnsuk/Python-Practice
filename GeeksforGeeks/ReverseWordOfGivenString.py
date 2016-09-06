#!/bin/bash

import sys

def reverseWordOfSentence(str):
    arr = str.split(" ")
    arr2 = []
    for st in arr:
        arr2.append(st[::-1])

    print " ".join(arr2)

str = "I like this program very much"
reverseWordOfSentence(str)
