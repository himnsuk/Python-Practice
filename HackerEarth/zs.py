def MaxOne(N, S):
	x = 0
	y = 1

	maxCount = 0
	while y < N:
		if (S[x] == "0" and S[y] == "0") or (S[x] == "1" and S[y] == "1") :
			maxCount+= 2
			x += 2
			y += 2
		else:
			maxCount += 1
			x += 2
			y += 2
	if x < N and y == N:
		if S[x] == "0":
			maxCount += 1
	return maxCount

print(MaxOne(2, "00"))
print(MaxOne(4, "0010"))
print(MaxOne(5, "00000"))