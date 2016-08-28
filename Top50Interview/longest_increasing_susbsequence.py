from typing import List

def LIS(nums: List[int]) -> List[int]:
	l = len(nums)
	i = 1
	j = 0

	temp: List[int] = [1]*l
	print(nums)
	while i < l:
		for j in range(i):
			if nums[j] < nums[i] and temp[i] < temp[j] + 1:
				temp[i] = temp[j] + 1
			j += 1
		i += 1

	print(temp)
	max_lis:int = temp[0]
	for x in temp:
		if x > max_lis:
			max_lis = x
	
	k = l
	while
	return max_lis

nums = [3,4,-1,0,6,2,3, 4]
# nums = [50, 3, 10, 7, 40, 80]
print(LIS(nums))
# %%
