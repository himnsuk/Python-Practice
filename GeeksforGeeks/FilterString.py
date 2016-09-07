#!/bin/bash

import sys

ONE = 1
TWO = 2

def filterString(string):
    state = ONE
    j = 0
    for i in xrange(len(string)):
        if state == ONE and string[i] != 'a' and strin[i] != 'b':
            string[j] = string[i]
            j += 1

        if state == TWO and string[i] != 'c':
            string[j] = 'a'
            j += 1
            if string[i] != 'a' and string[i] != 'b':
                string[j] = string[i]
                j += 1

        state = TWO if string[i] == 'a' else ONE
    if state == TWO:
        string[j] = 'a'
        j += 1

    return ''.join(string[:j])

def filterConventional(string):
    j = 0
    for i in xrange(len(string)):
        if string[i] != 'b':
            string[j] = string[i]
            j += 1
        elif string[i] != 'a' and string[i+1] != 'c':
            string[j]
print filterString(list('ababac'))
