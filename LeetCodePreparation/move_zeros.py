
def move_zeros(arr):
    zero = 0
    num = 0

    for i in range(len(arr)):
        if arr[i] == 0:
            num += 1
        else:
            arr[zero], arr[num] = arr[num], arr[zero]
            zero += 1
            num += 1
    
    return arr

arr = [0, 0, 0, 1, 0, 3, 0, 0, 12]
# arr = [0]

print(move_zeros(arr))