
def get_coefficient(n, k):
    if k < n-k:
        k = n-k
    
    res = 1
    for i in range(k):
        res *= n-i
        res //= i+1
    
    return res

print(get_coefficient(15,2))
