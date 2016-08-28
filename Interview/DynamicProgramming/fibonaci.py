
# Top Down Approach
def fibonacci(n, lookup):
    if n == 0 or n == 1:
        lookup[n] = n
    
    if lookup[n] is None:
        lookup[n] = fibonacci(n-1, lookup) + fibonacci(n-2, lookup)
    
    return lookup[n]

# Bottom up approach
def fibonacci2(n):
    lookup = [0]*(n + 1)

    lookup[1] = 1

    for x in range(2, n+1):
        lookup[x] = lookup[x-1] + lookup[x-2]
    
    return lookup[n]


def main():
    n = 34

    # lookup = [None]*(101)

    # print(fibonacci(n, lookup))
    # print(lookup)

    print(fibonacci2(n))

if __name__ == "__main__":
    main()