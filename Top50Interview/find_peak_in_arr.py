# Given a non-empty array of integers arr, create a function that
# returns the index of a peak element. We consider an element as
# peak if it's greater than or equal to its neighbors, the next and
# previous element, and the first element have at most one neighbor only. 
# And if there are multiple peaks in arr, just return the index of one of them

def findPeak(arr):
	a = 1
	b = len(arr) - 1
	peakList = []
	while (a < b):
		if arr[a] >= arr[a-1] and arr[a] >= arr[a+1]:
			peakList.append(a)
		a += 1
	return peakList
	

# arr = [2, 1, 3,2,5,8,7]
arr = [1, 5, 8, 8, 3, 9]
print(findPeak(arr))


# Given a linked list of integers list, create a boolean
# function that checks if it's  a palindrome linked list
# in constant space complexity

def reverseLinkedList(head):
	prev = None
	current = head
	while current:
		next = current.next
		current.next = prev
		prev = current
		current = next
	head = prev

def linkedListPalindrome(head):
	fast = slow = head:
	while fast:
		slow = slow.next
		fast = fast.next.next
	
	slow = reverseLinkedList(slow)
	first = head
	second = slow
	while second:
		if first.val != second.val:
			return False
		first = first.next
		second = second.next
	return True


# Given a string str made of alphabetical letters only,
# create a function that returns the length of the longest
# palindrome string that can be made from letters of str

def findLongestPalindrome(st):
	helpDict = {}
	for s in st:
		if s in helpDict:
			helpDict[s] += 1
		else:
			helpDict[s] = 1
	
	carry = 0
	total = 0
	for val in helpDict.values():
		if val%2 == 0:
			total += val
		elif val%2 != 0:
			if val > 1:
				total += val - 1
				carry = 1
			if carry == 0 and val == 1:
				carry = 1
	
	return total + carry

# st = "abababcabababe"
st = "abab"
print(findLongestPalindrome(st))
 
# Given two strings str1 and str2,
# create a function that returns the
# first index where we can find str2
# in str1. If we cannot find str2 in
# str1, the function must retur -1