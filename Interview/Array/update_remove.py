
# Write a program which takes as input an array of characters, and removes each 'b' and replaces
# each'a'by two 'd's. Specifically, along with the array, you are provided an integer-valued size. Size
# denotes the number of entries of the array that the operation is to be applied to. You do not have
# to worry about preserving subsequent entries. For example, if the array is (a,b,A,c,-) and the size
# is 4, then you c€ul retum (d,d,d,d,c). You can assume there is enough space in the array to hold the
# final result.


def update_remove(s):
    a_count, ui = 0, 0
    for x in range(len(s)):
        if s[x] != "b":
            s[ui] = s[x]
            ui += 1
        if s[x] == "a":
            a_count += 1

    for y in range(ui, len(s)):
        s[y] = ''

    if (ui+a_count) > len(s):
        print(a_count)
        s = s + [""]*((a_count + ui) - len(s))

    k = ui + a_count - 1
    ui = ui - 1

    while k > 0:
        if s[ui] == "a":
            s[k] = "d"
            s[k-1] = "d"
            k -= 2
            ui -= 1
        else:
            s[k] = s[ui]
            k -= 1
            ui -= 1
    return s


# a = "aabbccadefg"
a = "aabbccadefga"
b = list(a)
print(b)
print(update_remove(b))
