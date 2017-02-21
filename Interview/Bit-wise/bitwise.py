# Notice that bit-length, which is the number of binary digits, 
# varies greatly across the characters. The euro sign (€) requires fourteen bits, 
# while the rest of the characters can comfortably fit on seven bits.
# (42).bit_length()

# a = 5
# "{0:b}".format(a)
# b = a - 1

# "{0:b}".format(b)

# a^b
# a&b
# a|b

def add(set, num):
    if (set & ( 1 << (num - 1))) != (1 << (num - 1)):
        set = set ^ ( 1 << (num - 1))
    return set

def remove(set, number):
    set = set ^ (1 << (number - 1))
    return set

def display(set):
    for x in range(len(format(set, "b"))):
        if set & ( 1 << x):
             print(x + 1, end = " ")
    print()

if __name__ == "__main__":
    set = 15
    display(set)
    set = remove(set ,3)
    display(set)
    set = add(set, 5)
    display(set)
    set = add(set, 5)
    display(set)