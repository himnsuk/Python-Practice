#!/bin/bash

import sys

def fact(n):
    if n == 0 or n == 1:
        return 1
    else:
        return n*fact(n-1)

def lexicographic_ranking(arr):
    a = list(arr)
    M = len(a)
    total_rank = 0
    for i in range(M):
        rank = 0
        for j in range(i+1, M):
            if arr[i] > arr[j]:
                rank += 1
        
        total_rank += rank * fact(M-i-1)

    print(total_rank + 1)

arr = "STRING"
lexicographic_ranking(arr)


