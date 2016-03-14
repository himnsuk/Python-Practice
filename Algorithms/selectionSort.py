import sys

arr = [23, 8, 2, 17, 5]
def selectionSort(arr):
    n = len(arr)
    for i in range(n):
        idx = i
        for j in range(i, n):
            if arr[j] < arr[idx]:
                idx = j
        arr[i], arr[idx] = arr[idx], arr[i]

selectionSort(arr)
print arr
