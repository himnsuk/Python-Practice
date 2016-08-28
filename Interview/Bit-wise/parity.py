
# The below method complexity is O(n) as it will iterate
# through all the bits

# def parity(x):
#     result = 0
#     while x:
#         print(f"x value => {x}")
#         y = format(x, "b")
#         print(f"Binary x value => {y}")
#         result ^= x&1
#         print(f"result => {result}")
#         x >>= 1
#         print(f"x value after shifting right => {x}")
#     return result

def parity(x):
    result = 0
    while x:
        print(f"x value => {x}")
        y = format(x, "b")
        print(f"Binary x value => {y}")
        result ^= 1
        print(f"result => {result}")
        print(f"value of x-1 => {x-1}")
        y1 = format(x-1, "b")
        print(f"Binary x-1 value => {y1}")
        x &= x-1
        y2 = format(x, "b")
        print(f"Produced binary after and operator => {y2}")
        print(f"x value after shifting right => {x}")
    return result


z = 20
x = 21

print(f"Parity value of x => {parity(x)}")
# print(f"Parity value of z => {parity(z)}")

