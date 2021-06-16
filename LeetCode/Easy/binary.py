def searchInsert(nums, target):
    """
    :type nums: List[int]
    :type target: int
    :rtype: int
    """
    
    
    r, l = len(nums), 0
    if nums[r-1] < target:
        return r
    
    ans = -1
    while r >= l:
        m = (r+l)//2
        if nums[m] == target:
            return m
        elif nums[m] < target:
            ans, l = m + 1, m + 1
        else:
            r = m - 1
            
    if ans == -1:
        return 0
    return ans

nums = [1,3,5,7]
print(searchInsert(nums, 2))