# Input: nums = [-2,1,-3,4,-1,2,1,-5,4]
# Output: 6
# Explanation: [4,-1,2,1] has the largest sum = 6.

from typing import List


def maximumContinuosArraySum(arr: List[int]) -> List[int]:
	maximum = arr[0]
	total = 0
	
	for i in range(1, len(arr)-1):
		total = arr[i-1] + arr[i]

		if total > arr[i]:
			arr[i] = total

		if total > maximum:
			maximum = total
		
	return maximum

arr = [-2, 1, -3, 4, -1, 2, 1, -5, 4]
print(maximumContinuosArraySum(arr))
