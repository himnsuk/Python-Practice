def square_sorted(arr):
    left = 0
    n = len(arr)
    right = index = n-1
    result = [0]*n

    while left <= right:
        if abs(arr[left]) > abs(arr[right]):
            result[index] = arr[left]**2
            left += 1
        else:
            result[index] = arr[right]**2
            right -= 1
        index -= 1
    return result


arr = [-4,-1,0,3,10]
print(square_sorted(arr))