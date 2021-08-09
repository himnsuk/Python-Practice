
#%%
def lenghtOfLIS(nums):
	max_sub = 1

	i = 0
	j = 1

	subse = 1

	l = len(nums)
	if l == 0:
		return 0

	while j < l:
		if nums[i] < nums[j]:
			subse += 1
			j += 1
			i += 1
		else:
			i = j
			j += 1
			if subse > max_sub:
				max_sub = subse
			subse = 1
		
	if subse > max_sub:
		max_sub = subse

	return max_sub

nums = [3, 10, 3, 11, 4, 5, 6, 7, 8, 12, 3, 4, 5, 6, 7, 8, 9, 10]	
print(lenghtOfLIS(nums))

nums2 = []
print(lenghtOfLIS(nums2))

# %%
