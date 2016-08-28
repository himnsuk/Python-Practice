#!/bin/bash

import sys

def getChar(N, d):
    num = str(N)
    return num[d-1]

def getNthChar(N):
    sum = 0
    dist = 0
    nine = 9
    for i in range(1, N):
        dist += nine
        sum += nine*i

        if sum >= N:
            dist -= nine
            sum -= nine*i
            N -= sum
            break

        nine *= 10

    diff = int(N/i)

    d = (N)%i
    if d == 0:
        d = i

    print(getChar(diff + dist, d))


N = 251
getNthChar(N)
