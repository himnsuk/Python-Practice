def magical_string (s):
    # write your code here
    n = len(s)
    index_A, index_B = 0, 0
    count_A, count_B = 0, 0
    initial_B = 0
    for i in range(n):
      if s[i] == 'A':
        index_A = i
        count_A += 1
        if index_A > index_B:
          count_B = 0
      if s[i] == 'B':
        if count_A == 0:
          initial_B += 1
        else:
          index_B = i
          count_B += 1
    
    if count_A == 0 and count_B == 0:
      return "Impossible"

    magic_string =''

    if initial_B != 0:
      for i in range(initial_B):
        magic_string += 'B'

    if count_A != 0:
      for i in range(count_A):
        magic_string += 'A'

    if count_B != 0:
      for i in range(count_B):
        magic_string += 'B'
    
    mx = ""
    for i in range(len(magic_string)):
        mx = max(mx, magic_string[i:])
    return mx

T = int(input())
for _ in range(T):
    s = input()

    out_ = magical_string(s)
    print (out_)