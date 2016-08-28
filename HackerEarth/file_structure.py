#!bin/bash/

import sys


# def max_distance(arr):
#     simple_hash = {}
#     L = len(arr)
#
#     for i in range(L):
#         if arr[i] in simple_hash:
#             simple_hash[arr[i]].append(i)
#         else:
#             simple_hash[arr[i]] = []
#             simple_hash[arr[i]].append(i)
#
#     max_distance = 0
#     for key, value in simple_hash.iteritems():
#         distance = value[-1] - value[0]
#         if distance > max_distance:
#             max_distance = distance
#     if max_distance == 0:
#         print -1
#     else:
#         print max_distance

n = int(raw_input().strip())
simple_hash = {}
for i in range(n-1):
    arr = raw_input().strip().split(' ')
    if arr[0] == "root":
        simple_hash[arr[0]].append({arr[1]: [arr[)
    else:
        simple_hash[arr[0]] = []
        simple_hash[arr[0]].append(arr[1])

print simple_hash

query = []
m = int(raw_input().strip())
for j in range(m):
    a = raw_input().strip()
    query.append(a)

print query
