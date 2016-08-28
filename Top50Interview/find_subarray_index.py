
# Given two strings str1 and str2,
# create a function that returns the
# first index where we can find str2
# in str1. If we cannot find str2 in
# str1, the function must retur -1

# Lets first create brute-force solution


# def findSubArrayIndex(str1, str2):
#     if len(str1) < len(str2):
#         return -1

#     a = 0
#     b = 0
#     c = 0
#     found = False
#     while a < len(str1):
#         if str1[a] == str2[b]:
#             c = a + 1
#             b = 1
#             while b < len(str2):
#                 if str1[c] == str2[b]:
#                     c += 1
#                     b += 1
#                 else:
#                     b = 0
#                     break
#             if b == len(str2) and str1[c-1] == str2[b - 1]:
#                 found = True
#                 break
#         a += 1

#     return a if found else -1


# def findSubArrayIndex2(str1, str2):
#     if len(str1) < len(str2):
#         return -1

#     n = len(str1)
#     m = len(str2)

#     for i in range(n-m+1):
#         found = True
#         for j in range(m):
#             if str1[i+j] != str2[j]:
#                found = False
#                break
#         if found:
#             return i
#     return -1

# str1 = "aabbaababab"
# str2 = "aaba"

# print(findSubArrayIndex2(str1, str2))
def preprocessingSubString(str2):
    a = 1
    b = 0
    m = len(str2)
    lpsArr = [0]*m
    while a < m:
        if str2[a] == str2[b]:
            b += 1
            lpsArr[a] = b
            a += 1
        elif b > 0:
            b = lpsArr[b-1]
        else:
            lpsArr[a] = 0
            a += 1

    return lpsArr

def findSubArrayIndexKMP(str1, str2):
    m = len(str1)
    n = len(str2)
    if m < n: return -1
    if m == n and str1 == str2: return 0
    if str2 == "": return 0

    lpsArray = preprocessingSubString(str2)

    i = j = 0
    while i < m and j < n:
        if str1[i] == str2[j]:
            i += 1
            j += 1
        elif j > 0:
            j = lpsArray[j-1]
        else:
            i += 1
    return -1 if j < n else i-j




# str2 = "aaabaabaaaababa"
# print(preprocessingSubString(str2))


str1 = "abbaababab"
str2 = "aaba"
print(findSubArrayIndexKMP(str1, str2))


# Given a binary tree of integers root,
# create a function that returns an array
# where each element represents an array 
# that contains the elements at the level i

# binary level order traversal
# And print all level data in groups as array
class Node:
    def __init__(self, data) -> None:
        self.data = data
        self.left = None
        self.right = None

def levelOrderTraversal(root):
    if not root:
        return
    
    queue = []
    queue.append(root)

    levelOrder = []
    while queue:
        current = queue.pop(0)
        levelOrder.append(current.data)
        print(current.data, end=", ")
        if current.left: queue.append(current.left)
        if current.right: queue.append(current.right)
    
    return levelOrder

def levelOrderTraversalWithEachLevelTogether(root):
    if not root:
        return
    
    queue = []
    queue.append((root, 0))

    levelOrder = []
    while queue:
        current = queue.pop(0)
        if len(levelOrder) < current[1] + 1:
            levelOrder.append([current[0].data])
        else:
            levelOrder[current[1]].append(current[0].data)
        # print(current.data, end=", ")
        if current[0].left: queue.append((current[0].left, current[1] + 1))
        if current[0].right: queue.append((current[0].right, current[1] + 1))
    
    return levelOrder


# Root Node
root = Node(5)
# Left SubTree
root.left = Node(10)
root.left.left = Node(4)
root.left.right = Node(6)
root.left.left.right = Node(8)
root.left.right.left = Node(9)
root.left.right.right = Node(1)
# Right Sub Tree
root.right = Node(3)
root.right.right = Node(7)
root.right.right.left = Node(2)


# print(levelOrderTraversal(root))
print(levelOrderTraversalWithEachLevelTogether(root))


# Given a linked list, create a function that
# sorts it in ascending order. Note that the function
# returns nothing, the list must be sorted in-place
def sortLinkeListUsingMergeSort(head):
    first = head
    while first:
       second = head.next
       while second.next:
           if second.data > second.next.data:
               second.data, second.next.data = second.next.data, second.data
           second = second.next
       first = first.next
    return head

def mergeList(left, right):
    result = None

    if left == None:
        return right
    if right == None:
        return left
    
    if left.data <= right.data:
        result = left
        result.next = mergeList(left.next, right)
    else:
        result = right
        result.next = mergeList(left, right.next)
    return result

def mergeSort(head):
    if head == None or head.next == None:
        return head
    
    middle = getMiddle(head)
    nextToMiddle = middle.next
    middle.next = None
    left = mergeSort(head)
    right = mergeSort(nextToMiddle)
    result =  mergeList(left, right)
    return result

def getMiddle(head):
    if head == None:
        return head
    slow = head
    fast = head
    while fast.next != None and fast.next.next != None:
        slow = slow.next
        fast = fast.next.next
    return slow

class LinkedList:
    def __init__(self) -> None:
        self.head = None
 
    def printList(self):
        node = next
        while node:
            print(node.data, end=" -> ")
class Node:
    def __init__(self, data) -> None:
        self.next = None
        self.data = data


ll = LinkedList().head
ll = Node(5)
ll.next = Node(8)
ll.next.next = Node(4)
ll.next.next.next = Node(3)
ll.next.next.next.next = Node(11)
ll.next.next.next.next.next = Node(1)
        
# node = sortLinkeListUsingMergeSort(head)
node = mergeSort(ll)

current = node
while current:
    print(current.data, end = " -> ")
    current = current.next


# Given a binary find the height of it
class Node:
    def __init__(self, data) -> None:
       self.left = None
       self.right = None
       self.data = data 


def treeHeight(root):
    if not root:
        return 0
    
    left = treeHeight(root.left) + 1
    right = treeHeight(root.right) + 1

    maximum = left
    if right > left:
       maximum = right
    
    return maximum

# Given a binary tree find if the tree is balanced
def isTreeBalance(root):
    if not root:
        return True
    
    leftTreeHeight = treeHeight(root.left)
    rightTreeHeight = treeHeight(root.right)

    if abs(leftTreeHeight - rightTreeHeight) >= 2:
        return False

    return True
# root of binary tree
root = Node(4)

# Left sub-tree
root.left = Node(2)
root.left.left = Node(1)
root.left.right = Node(3)
# Right sub-tree
root.right = Node(6)
root.right.left = Node(5)
root.right.right = Node(7)
root.right.right.right = Node(9)
# root.right.right.right.right = Node(12)
# root.right.right.right.left = Node(11)
# root.right.right.right.left.left = Node(10)


# print(treeHeight(root))
print(isTreeBalance(root))


# Given a m x n grid filled with non-negative numbers, 
# find a path from top left to bottom right, 
# which minimizes the sum of all numbers along its path.

# Note: You can only move either down or right at any point in time.

#%%
def findMinimum(list, m, n):
    row = len(list)
    col = len(list[0])

    if m == row-1 and n == col-1:
        return list[m][n]
    
    elif m < row - 1 and n < col - 1:
        return min(findMinimum(list, m + 1, n), findMinimum(list,m, n+1)) + list[m][n]

    elif m < row - 1: 
        return findMinimum(list, m + 1, n) + list[m][n]
    elif n < col - 1: 
        return findMinimum(list, m , n + 1) + list[m][n]

    
def findMinimumDynamic(list):
    row = len(list)
    col = len(list[0])
    newList = [[0 for x in range(col)] for y in range(row)]

    newList[0][0] = list[0][0]
    for i in range(row):
        for j in range(col):
            if i == 0 and j != 0:
                newList[i][j] = list[i][j] + newList[i][j-1]
            elif j == 0 and i != 0:
                newList[i][j] = list[i][j] + newList[i-1][j]
            elif i < row and j < col:
                newList[i][j] = list[i][j] + min(newList[i-1][j], newList[i][j-1])
    
    print("Minimum sum path")

    for i in range(row):
        for j in range(col):
            print(newList[i][j], end=", ")
        print()

    print()
    i = row - 1
    j = col - 1

    print("Path in the matrix => ", end=" ")
    while i >= 0 and j >= 0:
        print(f"{list[i][j]} ({i}, {j})", end=", ")
        if i == 0 and j >= 0:
            j -= 1
        elif j == 0 and i >= 0:
            i -= 1
        else:
            if newList[i-1][j] < newList[i][j-1]:
                i -= 1
            else:
                j -= 1
    
    print()

    return newList[row - 1][col - 1]


arr = [
       [1, 3, 1], 
       [1, 5, 1], 
       [4, 2, 1]
       ]

# print(findMinimum(arr, 0, 0))
print(f"Minimum Sum Path =>  {findMinimumDynamic(arr)}")
#%%
# Given a set of non-negative integers, and a value sum, 
# determine if there is a subset of the given set with sum equal to given sum. 

# Example: 

# Input: set[] = {3, 34, 4, 12, 5, 2}, sum = 9
# Output: True  
# There is a subset (4, 5) with sum 9.

# Input: set[] = {3, 34, 4, 12, 5, 2}, sum = 30
# Output: False
# There is no subset that add up to 30.


def subsetSumRecursive(arr, k):
    