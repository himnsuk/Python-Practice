import sys

def createHash():
    string = "littlejhool"
    M = len(string)
    hash_value = {}
    for i in range(M):
        hash_value[string[i]] = i
    return hash_value

def matched_girls(arr, girls):
    given_hash = createHash()
    N = arr[0]
    K = arr[1]
    new_hash = {}
    for girl in girls:
        count = 0
        girl = ''.join(list(set(girl)))
        for j in girl:
            if j in given_hash:
                count += 1
        new_hash[count] = girl
        print new_hash

# def requiredHash(string):
#     given_hash = createHash()
#     M = len(string)
#     sum = 0
#     for i in range(M):
#         n = len(string[i])
#         curr_str = string[i]
#         for j in range(n):
#             sum += (j + given_hash[curr_str[j]])
#
#     grand_total.append(sum*M)
#
n = int(raw_input().strip())

for i in range(n):
    arr = map(int, raw_input().strip().split(' '))
    girls = raw_input().strip().split(' ')
    matched_girls(arr, girls)



