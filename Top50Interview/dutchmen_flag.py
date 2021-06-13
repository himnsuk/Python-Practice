
# Given arr= [2,0,0, 1, 2,1, 0, 0]

def dutchmenFlag(arr):
    a = 0
    b = 0
    c = len(arr) - 1

    while b <= c:
        if arr[b] == 1:
            b += 1
        elif arr[b] == 0 and arr[a] == 0:
            a += 1
            b += 1
        elif arr[b] == 0 and arr[a] == 1:
            arr[a], arr[b] = arr[b], arr[a]
            a += 1
            b += 1
        elif arr[b] == 2 and arr[c] == 1:
            arr[b], arr[c] = arr[c], arr[b]
            b += 1
            c -= 1
        elif arr[b] == 2 and arr[c] == 0:
            arr[b], arr[c] = arr[c], arr[b]
            c -= 1
    return arr

def dutchmen_flag2(arr):
	low = 0
	mid = 0
	high = len(arr) - 1

	while mid <= high:
		if arr[mid] == 0:
			arr[mid], arr[low] = arr[low], arr[mid]
			mid += 1
			low += 1
		elif arr[mid] == 1:
			mid += 1
		elif arr[mid] == 2:
			arr[mid], arr[high] = arr[high], arr[mid]
			high -= 1
	
	return arr

# 0 - [2,0,0, 1, 2,1, 0, 0]
# 1 - [0,0,0, 1, 2,1, 0, 2]
# 2 - [0,0,0, 1, 0,1, 2, 2]


arr = [2, 1, 0, 1, 2, 1, 0, 0]
# arr = [2, 1, 0, 1, 2, 0, 0, 1]
# print(dutchmenFlag(arr))
print(dutchmen_flag2(arr))

# 0 = [2, 1, 0, 1, 2, 1, 0, 0] c = 7, a = 0, b = 0
# 1 = [0, 1, 0, 1, 2, 1, 0, 2] c = 6, a = 0, b = 0
# 2 = [0, 1, 0, 1, 2, 1, 0, 2] c = 6, a = 1, b = 1
# 3 = [0, 1, 0, 1, 2, 1, 0, 2] c = 6, a = 1, b = 2
# 4 = [0, 0, 1, 1, 2, 1, 0, 2] c = 6, a = 2, b = 3
# 5 = [0, 0, 1, 1, 2, 1, 0, 2] c = 6, a = 2, b = 4
# 6 = [0, 0, 1, 1, 0, 1, 2, 2] c = 5, a = 2, b = 4
# 7 = [0, 0, 0, 1, 1, 1, 2, 2] c = 5, a = 3, b = 5
# 8 = [0, 0, 0, 1, 1, 1, 2, 2] c = 5, a = 3, b = 5