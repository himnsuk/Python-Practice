#!/bin/bash

import sys

def printArrayWithGivenString(str1, arr):
    n = len(str1)
    count = [0]*256
    for i in str1:
       count[ord(i)] = 1
    for st in arr:
        m = len(st)
        if m < n:
            return
        cnt = 0
        for j in st:
            if count[ord(j)] == 1:
               cnt += 1
               count[ord(j)] = 0
        if cnt == n:
           print st
        for i in str1:
           count[ord(i)] = 1
str1 = "sun"
arr = ["geeksforgeeks", "unsorted", "sunday", "just", "sss"]
printArrayWithGivenString(str1, arr)
