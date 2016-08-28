# Input: x = 123
# Output: 321

## First Iteration
# def reverse_integer(x):
#     """
#     :type x: int
#     :rtype: int
#     """
#     if x <= (-2**31) or x >= (2**31) - 1:
#         return 0
#     sign = 1
#     if x < 0:
#         sign = -1
#         x = -1*x
#     store = []
#     while(x != 0):
#         store.append(x % 10)
#         x = x//10

#     reverse_num = 0
#     for ind, y in enumerate(reversed(store)):
#         reverse_num += y*(10**ind)

#     return reverse_num*sign


# print(reverse_integer(1534236469))

# Second Iteration
def reverse_integer(x):
    """
    :type x: int
    :rtype: int
    """
    if x <= -2147483648 or x >= 2147483647:
        return 0
    sign = 1
    if x < 0:
        sign = -1
        x = -1*x
    ans = 0
    while(x != 0):
        ans *= 10
        ans += x % 10
        x = x//10

    # reverse_num = 0
    # for ind, y in enumerate(reversed(store)):
    #     reverse_num += y*(10**ind)
    if ans >= 2147483647 or ans <= -2147483648:
        return 0
    return ans*sign


print(reverse_integer(1534236469))
print(reverse_integer(-1453))
