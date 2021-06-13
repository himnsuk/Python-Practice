# Floyd cycle detection algorithm
# arr = [5,2,4,2,1,6,3]
def findDuplicate(arr):
	tortoise = arr[0]
	hare = arr[arr[0]]
	while tortoise != hare:
		tortoise = arr[tortoise]
		hare = arr[arr[hare]]
	tortoise = 0

	while tortoise != hare:
		tortoise = arr[tortoise]
		hare = arr[hare]
	return tortoise

arr = [5,2,4,2,1,6,3]

print(findDuplicate(arr))