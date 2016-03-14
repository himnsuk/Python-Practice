import sys


n,t = raw_input().strip().split(' ')
n,t = [int(n),int(t)]
width = map(int,raw_input().strip().split(' '))
for a0 in xrange(t):
    i,j = raw_input().strip().split(' ')
    i,j = [int(i),int(j)]
    min = width[i]
    for m in xrange(i+1,j+1):
        if min > width[m]:
            min = width[m]

    print min
