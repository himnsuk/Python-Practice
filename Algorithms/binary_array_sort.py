#!/bin/bash

import sys

def sort_binary(arr):
    M = len(arr)

    l = -1

    for j in range(M):
        if arr[j] < 1:
            l += 1
            arr[j], arr[l] = arr[l], arr[j]

    print(arr)

arr = [0, 1, 1, 1, 0, 0, 1, 0]

sort_binary(arr)
