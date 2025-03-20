

def dutchmen_flag(arr):
    l = 0
    m = 0
    r = len(arr) - 1

    while m <= r:
        if arr[m] == 0:
            arr[l], arr[m] = arr[m], arr[l]

            l += 1
            m += 1
        elif arr[m] == 1:
            m += 1
        elif arr[m] == 2:
            arr[m], arr[r] = arr[r], arr[m]
            r -= 1
    
    return arr

arr = [1,2, 0, 0, 1,2, 0, 1]

if __name__ == "__main__":
    print(dutchmen_flag(arr))