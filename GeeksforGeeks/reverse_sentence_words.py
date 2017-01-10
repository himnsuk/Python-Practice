#!/bin/bash

import sys

def reverseWord(arr, start, end):
    while start < end:
        arr[start], arr[end] = arr[end], arr[start]
        start += 1
        end -= 1

def reverseString(arr):
    start = 0
    M = len(arr)
    for i in range(M):
        if i == M-1:
            end = i
            reverseWord(arr, start, end)

        elif arr[i] == ' ':
            reverseWord(arr, start, i-1)
            start = i+1


    reverseWord(arr, 0, M-1)

    print(''.join(arr))


st = "I love Programming"

arr = list(st)

reverseString(arr)
