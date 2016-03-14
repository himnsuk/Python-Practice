#!/bin/python

import sys
n = int(raw_input().strip())
arr = map(int,raw_input().strip().split(' '))
arr.sort()
arr_length = len(arr)
while arr_length > 1:
    print arr_length
    remain_arr = map(lambda x: x-arr[0], arr)
    get_arr = filter(lambda x: x!=0, remain_arr)
    arr_length = len(get_arr)
    arr = get_arr
