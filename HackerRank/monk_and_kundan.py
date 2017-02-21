import sys

def createHash():
    string = "abcdefghijklmnopqrstuvwxyz1234567890ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    M = len(string)
    hash_value = {}
    for i in range(M):
        hash_value[string[i]] = i

    return hash_value

def requiredHash(string):
    given_hash = createHash()
    M = len(string)
    sum = 0
    for i in range(M):
        n = len(string[i])
        curr_str = string[i]
        for j in range(n):
            sum += (j + given_hash[curr_str[j]])

    grand_total.append(sum*M)

grand_total = []
n = int(raw_input().strip())

for i in range(n):
    arr = raw_input().strip().split(' ')
    requiredHash(arr)

for k in grand_total:
    print k
