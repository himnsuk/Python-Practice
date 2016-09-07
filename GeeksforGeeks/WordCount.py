#!/bin/bash

import sys

def wordCount(string):
    n = len(string)
    str2 = string.replace("\n", " ").replace("\t", " ").split()
    print len(str2)
    print str2
    print string

string = "One two          three\n  four\nfive  "

wordCount(string)
