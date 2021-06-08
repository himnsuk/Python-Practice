
#!/bin/bash

import sys
import math
def solution(A, B):
    As = str(A)
    Bs = str(B)

    L1 = len(As)
    L2 = len(Bs)
    Cs = ""
    i = 0
    j = 0

    while(i < L1 and j < L2):
        Cs += As[i]
        Cs += Bs[j]
        i += 1
        j += 1

    while(i < L1):
        Cs += As[i]
        i += 1

    while(j < L2):
        Cs += Bs[j]
        j += 1

    print int(Cs)

solution(123, 67890)
