import string

a = "1234"
def string_to_int(s):
    result = 0
    for i, v in enumerate(reversed(s)):
        result += 10**i * string.digits.index(v)
    return result 

print(string_to_int(a))