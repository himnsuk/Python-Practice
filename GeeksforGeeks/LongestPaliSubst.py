#!/bin/bash

import sys

def longestPaliSub(String):
    n = len(String)
    table = []
    for i in range(n):
        new = []
        for j in range(n):
            new.append(False)
        table.append(new)

    max_length = 1
    for i in range(n):
        table[i][i] = True

    start = 0
    for i in range(n-1):
        if(String[i] == String[i+1]):
            table[i][i+1] = True
            start = i
            max_length = 2

    for k in range(3, n+1):
        for i in range(n-k+1):
            j = i+k-1
            if table[i+1][j-1] and String[i] == String[j]:
                table[i][j] = True
                if k > max_length:
                    start = i
                    max_length = k
    return max_length

String = "geeg"

print longestPaliSub(String)
