#! /bin/bash

import sys

def isInterleaving(A, B, C):
    M = len(C)
    print(M)
    i = 0
    j = 0
    k = 0

    while k != M-1:
        if A[i] == C[k]:
            i += 1

        elif B[j] == C[k]:
            j += 1

        else:
            return 0
        k += 1
    if A[i - 1] or B[j - 1]:
        return 0
    return 1

A = "AB"
B = "CD"
C = "ABCD"

print(isInterleaving(A, B, C))

if A in C and B in C:
    print("Interleaving Strings")
