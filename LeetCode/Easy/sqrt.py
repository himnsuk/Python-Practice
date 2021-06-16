def mySqrt(x):
    left = 0
    right = x
    mid = 0

    while left< right:
        mid = right-(right-left)//2

        if mid<=(x//mid):
            left = mid
        else:
            right= mid-1
    return left


print(mySqrt(5))