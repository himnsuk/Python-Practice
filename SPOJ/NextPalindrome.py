#!/bin/python

import sys

def AreAll9s(num_arr):
    n = len(num_arr)
    for i in num_arr:
        if i != 9:
            return 0
    return 1
def checkPalindrom(arr):
    n = len(arr)
    if len(arr)%2 == 0:
        i = len(arr)/2 - 1
        j = len(arr)/2
        while(i >= 0 and j < n):
            if(arr[i] != arr[j]):
                return 0
            else:
               i = i - 1
               j = j + 1
        return 1
    else:
        i = len(arr)/2 - 1
        j = len(arr)/2 + 1
        while(i >= 0 and j < n):
            if(arr[i] != arr[j]):
                return 0
            else:
               i = i - 1
               j = j + 1
        return 1

n = int(raw_input("Enter Number of Test Cases ?").strip())

for i in range(n):
    num_arr = map(int, list(raw_input("Enter the number ?")))
    if checkPalindrom(num_arr):
        n = len(num_arr)
        if n%2 == 0:
            i = len(num_arr)/2 - 1
            j = len(num_arr)/2
            num_arr[i] = num_arr[i] + 1
            num_arr[j] = num_arr[j] + 1
            print ''.join(map(str, num_arr))
        else:
            mid = len(num_arr)/2 
            num_arr[mid] = num_arr[mid] + 1
            print ''.join(map(str, num_arr))

    else:
        n = len(num_arr)
        if n%2 == 0:
            i = len(num_arr)/2 - 1
            j = len(num_arr)/2
            num_arr[i] = num_arr[i] + 1
            while(i >= 0):
                num_arr[j] = num_arr[i]
                i = i - 1
                j = j + 1
        else:
            i = len(num_arr)/2 
            j = len(num_arr)/2 + 1
            num_arr[i] = num_arr[i] + 1
            i = i - 1
            while(i >= 0):
                num_arr[j] = num_arr[i]
                i = i - 1
                j = j + 1
            print ''.join(map(str, num_arr))
