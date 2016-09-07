#!/bin/bash

import sys

def diviideEqualPart(str1, divide_size):
    n = len(str1)
    if n%divide_size != 0:
        print "Invalid input string"
        return
    k = 0
    for i in str1:
        if k%divide_size == 0:
            print "\n"
        print i
        k += 1

str1 = "a_simple_divide_string_quest"
b = 4
diviideEqualPart(str1, b)
