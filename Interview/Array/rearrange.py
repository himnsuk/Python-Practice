
def rearrange(A):
    for i in range(len(A)):
        A[i:i+2] = sorted(A[i:i+2], reverse = i%2)

    return A


A = [i for i in range(1,11)]

print(rearrange(A))



# Output => [1, 3, 2, 5, 4, 7, 6, 9, 8, 10]