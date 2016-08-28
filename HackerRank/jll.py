def solve (n, A):
    # Write your code here
    assist = {}
    for i in range(n):
      if A[i] not in assist:
        assist[A[i]] = [i]
      else:
        assist[A[i]].append(i)
    for i in range(n):
      if len(assist[A[i]]) == 1:
        A[i] = -1
      else:
        for ind, x in enumerate(assist[A[i]]):
          if i == x and ind == len(assist[A[i]]) - 1:
            A[i] = abs(assist[A[i]][-1] - assist[A[i]][-2])
            break
          elif i == x:
            print(assist[A[i]])
            A[i] == abs(assist[A[i]][ind] - assist[A[i]][ind+1])
            break

    return A
    pass
    

n = int(input())
A = list(map(int, input().split()))

out_ = solve(n, A)
print (' '.join(map(str, out_)))