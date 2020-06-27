#!/bin/python

import sys

n = int(raw_input().strip())
arr = []
for i in range(n):
    arr.append(raw_input().strip())

def winner(arr):
    dicti = {}
    max = 0

    for i in range(n):
        if arr[i] in dicti:
            dicti[arr[i]] += 1
            if dicti[arr[i]] > max:
                max = dicti[arr[i]]
        else:
            dicti[arr[i]] = 0

    arr2 = []
    for key, value in dicti.iteritems():
        if value == max:
            arr2.append(key)


    arr2 = sorted(arr2, reverse = True)
    print arr2[0]
    return arr2[0]


winner(arr)
