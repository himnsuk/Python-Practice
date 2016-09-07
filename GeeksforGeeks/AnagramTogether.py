#!/bin/bash

import sys

class Word(object):
    def __init__(self, string, index):
        self.string = string
        self.index = index


def createDuplicateArray(string, size):
    dupArray = []
    for i in xrange(size):
        dupArray.append(Word(string[i], i))

    return dupArray

def PrintAnagramTogether(wordArr, size):
    dupArray = createDuplicateArray(wordArr, size)
    for i in xrange(size):
        dupArray[i].string = ''.join(sorted(dupArray[i].string))

    dupArray = sorted(dupArray, key=lambda k: k.string)
    for word in dupArray:
        print wordArr[word.index]

wordArr = ["cat", "dog", "tac", "god", "act"]
size = len(wordArr)
PrintAnagramTogether(wordArr, size)
