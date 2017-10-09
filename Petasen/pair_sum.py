n = int(raw_input().strip())

arr = map(int, raw_input().strip().split(' '))

L = len(arr)

dicti = {}

for i in range(L):
    dicti[arr[i]] = 0
count = 0
for i in range(L):
    j = i + 1
    while( j < L):
        if (arr[i] + arr[j]) in dicti:
            count += 1
        j += 1

print count
