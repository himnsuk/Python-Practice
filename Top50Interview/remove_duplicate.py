from typing import ValuesView


# Given an array of integers arr,
# create a function that returns
# an array that contains the values
# of arr without duplicates(the order doesn't matter)


def removeDuplicate(arr):
	arr.sort()

	a = 0
	b = 1

	while b < len(arr):
		if arr[a] == arr[b]:
			b += 1
		elif arr[a] != arr[b]:
			arr[a + 1], arr[b] = arr[b], arr[a + 1]
			a += 1
			b += 1
	return arr[:a+1]

arr = [2,3,2,5,3,4,9,3,2]
print(removeDuplicate(arr))