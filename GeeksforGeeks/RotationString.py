#!/bin/bash

import sys

def rotationString(str1, str2):
    comb_str = str1 + str1
    return str2 in comb_str

str1 = "ABCDE"
str2 = "CDABE"

print rotationString(str1, str2)
