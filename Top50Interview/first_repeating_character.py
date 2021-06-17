# Given a string str, create a function
# that returns the first repeating character
# (the first character that we have seen before).
# If such a chararcter doesn't exist, returnthe null

def first_repeat_character(st):
	temp_dict = {}

	for c in st:
		if c in temp_dict:
			return c
		else:
			temp_dict[c] = 1
	return None


st = "hema"
print(first_repeat_character(st))