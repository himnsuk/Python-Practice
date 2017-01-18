#!bin/bash/

import sys


def max_distance(arr):
    simple_hash = {}
    L = len(arr)

    for i in range(L):
        if arr[i] in simple_hash:
            simple_hash[arr[i]].append(i)
        else:
            simple_hash[arr[i]] = []
            simple_hash[arr[i]].append(i)

    max_distance = 0
    for key, value in simple_hash.iteritems():
        distance = value[-1] - value[0]
        if distance > max_distance:
            max_distance = distance
    if max_distance == 0:
        print -1
    else:
        print max_distance

n = int(raw_input().strip())
arr = map(int, raw_input().strip().split(' '))
max_distance(arr)
