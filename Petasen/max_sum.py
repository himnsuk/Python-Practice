n = int(raw_input().strip())

arr = map(int, raw_input().strip().split(' '))

def compare(a,b):
    str1 = str(a) + str(b)
    str2 = str(b) + str(a)
    if (int(str1) < int(str2)):
        return 1
    else:
        return -1

sorted(arr, cmp=compare)
str = ""
for j in reversed(arr):
    str += j

print str
