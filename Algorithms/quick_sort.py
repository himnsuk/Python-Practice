#!/bin/bash

import sys
import pdb

def quick_sort(arr, l, r):
    if l < r:
        pivot = partition(arr, l, r)
        quick_sort(arr, l, pivot-1)
        quick_sort(arr, pivot + 1, r)

def partition(arr, l, r):
    pivot = r

    left = l - 1
    for j in range(l, r):
        if arr[j] <= arr[pivot]:
            left += 1
            arr[left], arr[j] = arr[j], arr[left]


    arr[left + 1], arr[pivot] = arr[pivot], arr[left+1]
    return (left +  1)

arr = [3, 8, 5, 6, 4]

l = 0
r = len(arr)
print(arr)
quick_sort(arr, l, r-1)
print(arr)

