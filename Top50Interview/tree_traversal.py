class Node:
    def __init__(self, key) -> None:
        self.left = None
        self.right = None
        self.data = key


root = Node(4)

# Left Sub Tree
root.left = Node(3)
root.left.left = Node(1)
root.left.right = Node(2)
# Right Sub tree
root.right = Node(6)
root.right.left = Node(5)
root.right.right = Node(7)


# Tree Traversal
# 1. Pre-Order
def preOrderTraversal(root):
	if root == None:
		return
	print(root.data, end = ",")
	preOrderTraversal(root.left)
	preOrderTraversal(root.right)

# Pre-Order Traversal
# 4,3,1,2,6,5,7,
def preOrderTraversalIterative(root):
    stack = []
    stack.append(root)
    while stack:
        node = stack.pop()
        print(node.data, end=", ")
        if node.right:
            stack.append(node.right)
        if node.left:
            stack.append(node.left)

# print("Pre-Order Traversal")
# preOrderTraversalIterative(root)
# # 2. In-Order
# def inOrderTraversal(root):
# 	if root == None:
# 		return
# 	inOrderTraversal(root.left)
# 	print(root.data, end = ",")
# 	inOrderTraversal(root.right)
# print()
# print("In-Order Traversal")
# inOrderTraversal(root)
# In-Order Traversal
# 1,3,2,4,5,6,7,


def inOrderTraversalIterative(root):
    stack = []
    stack.append(root)
    current = root.left
    while True:
        if current:
            stack.append(current)
            current = current.left
        elif(stack):
            current = stack.pop()
            print(current.data, end=", ")
            current = current.right
        else:
            break



# With 2 stack
def postOrderTraversalIterativeWithTwoStack(root):
	st1 = []
	st2 = []
	st1.append(root)
	while st1:
		current = st1.pop()
		st2.append(current)
		if current.left:
			st1.append(current.left)
		if current.right:
			st1.append(current.right)
	
	while st2:
		print(st2.pop().data, end=", ")

# With one stack
def postOrderTraversalIterativeWithOneStack(root):
	st1 = []
	st1.append(root)
	current = root.left
	while True:
		if current:
			st1.append(current)
			current = current.left
		elif st1:
			current = st1.pop()
			if current.right:
				temp = current.right
				current.right = None
				st1.append(current)
				current = temp
			else:
				print(current.data, end=", ")
				current = current.right
		else:
			break


# # 3. Post-Order

def postOrderTraversal(root):
	if root == None:
		return
	postOrderTraversal(root.left)
	postOrderTraversal(root.right)
	print(root.data, end = ",")
print()
print("Post-Order Traversal")
postOrderTraversal(root)
