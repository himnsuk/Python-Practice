p = 4
spiral_walk = []
sm = [[p*row+j+1 for j in range(p)] for row in range(p)] 

def spiral_ordering(sm, offset):
    n = len(sm)
    for i in range(offset, n-offset):
        spiral_walk.append(sm[offset][i])

    for j in range(offset + 1, n-offset):
        spiral_walk.append((sm[j][n-offset-1]))
    
    for k in reversed(range(offset, n-offset-1)):
        spiral_walk.append(sm[n-offset-1][k])
    
    for l in reversed(range(offset + 1, n-offset - 1)):
        spiral_walk.append(sm[l][offset])

for i in range(int(len(sm)+1//2)):
    spiral_ordering(sm, i)

print(sm)
print(spiral_walk)