#!/bin/bash

import sys

p = 256

def printDuplicate(str):
    count = [0]*p

    for i in str:
        count[ord(i)] += 1

    for i in str:
        if count[ord(i)] > 1:
            print i ,count[ord(i)]
            count[ord(i)] = 0

str = "geeksforgeeks"
printDuplicate(str)
