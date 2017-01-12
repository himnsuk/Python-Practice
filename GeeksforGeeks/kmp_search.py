#!/bin/bash

import sys

def KMPSearch(pat, txt):
    M = len(pat)
    N = len(txt)
    lps = [0]*M

    createLPS(pat, M, lps)

    i = 0 #indexing in txt
    j = 0 #indexing in pat

    while i < N:
        if pat[j] == txt[i]:
            i += 1
            j += 1
        if j == M:
            print("Patter found in the string")
            j = lps[j-1]
        elif i < N and pat[j] != txt[i]:
            if j != 0:
                j = lps[j-1]
            else:
                i += 1
def createLPS(pat, M, lps):
   i = 1
   j = 0
   while i < M:
       if pat[i] == pat[j]:
           j += 1
           lps[i] = j
           i += 1
       else:
           if j != 0:
               j = lps[j-1]
           else:
               lps[i] = 0
               i +=1

txt = "ABABDABACDABABCABAB"
pat = "AB"
KMPSearch(pat, txt)
