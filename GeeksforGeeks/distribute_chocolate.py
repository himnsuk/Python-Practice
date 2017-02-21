#!/bin/bash

import sys

def distributeChocolate(arr, m):
    arr.sort()
    M = len(arr)
    if M < m:
        return 0

    first = 0
    last = m - 1

    min_diff = arr[last] - arr[first]

    while last < M:
        if (arr[last] - arr[first] < min_diff):
            min_diff = arr[last] - arr[first]

        first += 1
        last += 1

    print(min_diff)

# arr = [12, 4, 7, 9, 2, 23, 25, 41,30, 40, 28, 42, 30, 44, 48, 43, 50]
arr = [7, 3, 2, 4, 9, 12, 56]
m = 3
distributeChocolate(arr, m)
