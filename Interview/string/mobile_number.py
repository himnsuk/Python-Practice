#!/bin/bash

import sys
import pdb
import operator

keyboard = {
            'a': 2,
            'b': 2,
            'c': 2,
            'd': 3,
            'e': 3,
            'f': 3,
            'g': 4,
            'h': 4,
            'i': 4,
            'j': 5,
            'k': 5,
            'l': 5,
            'm': 6,
            'n': 6,
            'o': 6,
            'p': 7,
            'q': 7,
            'r': 7,
            's': 7,
            't': 8,
            'u': 8,
            'v': 8,
            'w': 9,
            'x': 9,
            'y': 9,
            'z': 9
           }

n = int(input("Enter number of string: "))
a = []
for i in range(n):
    a.append(input().strip())

a.sort()
new_dict = {}
for j in a:
    l = list(j)
    d = []
    for k in l:
        d.append(keyboard[k])

    new_dict[j] = int(''.join(map(str, d)))

sorte_x = sorted(new_dict.items(), key=operator.itemgetter(1), reverse=True)

for key, value in sorte_x:
    print(value, key)
