
# Write a program which takes as input a sorted atay A of integers and a positiveinteger m,
# and updates A so that if x appears z times in A it appears exactly mn(Z,m) times in A. The update
# to A should be performed in one pass, and no additional storage may be allocated.

def update_array(A, m):
    if not A:
        return 0
    
    mi = 0
    j = 0
    count = 1

    for i in range(1, len(A)):
        