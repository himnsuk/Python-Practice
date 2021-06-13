# Given a string str made of alphabetical letters only,
# create a function that returns the length of longest
# substring without repeating characters

def longestSubstringWithoutRepeat(st):
	a = 0
	b = 0
	temp = [False] * 128
	maximum = 0
	result = ""
	while b < len(st):
		if not temp[ord(st[b])]:
			temp[ord(st[b])] = True
			maximum = (b-a+1) if maximum < (b-a+1) else maximum
			b += 1
		elif a <= b:
			if st[a] == st[b]:
				b += 1
			a += 1
	
	return maximum

# st = "himanshu"
# st = "alphabetical"
st = "returns"
print(longestSubstringWithoutRepeat(st))
