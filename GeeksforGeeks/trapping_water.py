#!/bin/bash

import sys

def max(a, b):
   return a if a > b else b

def min(a, b):
   return a if a < b else b

def trapped_water(arr):
    left = [0]*len(arr)
    right = [0]*len(arr)

    N = len(arr)
    i = 1
    left[0] = arr[0]
    while i < N:
        left[i] = max(arr[i], left[i-1])
        i += 1

    right[N-1] = arr[N-1]
    i = N-2
    while i >= 0:
        right[i] = max(arr[i], right[i+1])
        i -= 1

    water = 0
    j = 0
    while j < N:
        water += (min(left[j], right[j]) - arr[j])
        j += 1

    print(arr)
    print(left)
    print(right)
    print(water)

arr = [3, 0, 0, 2, 0, 4]
trapped_water(arr)
