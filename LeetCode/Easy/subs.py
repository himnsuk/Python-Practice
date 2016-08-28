# def find(n):
#     total = 0
#     i = 0
#     while total < n:
#         i+=1
#         total += 2*int(i**(1/2))
#     return i

# print(find(6))



import math
def solve (Q, query):
    temp = []
    for q in range(Q):
        a = query[q][0]
        b = query[q][1]
        a_num = find(a)
        b_num = find(b)
        temp.append(b_num - a_num + 1)
    return temp

    # Write your code here
    pass
def find(n):
    total = 0
    i = int((n*2)**(1/2))
    # while total < n:
    #     i += 1
    #     total += 2*int(i**(1/2))
    return i


Q = int(input())
query = [list(map(int, input().split())) for i in range(Q)]

out_ = solve(Q, query)
print (' '.join(map(str, out_)))