# Given a binary tree of integers root,
# create a function that reverse it left
# to right in place

		#   	 4                4
		# 	    / \              / \
		#      2	6           6	2
		#     / \  / \         / \  / \
		#    1	 3 5  7        7  5 3  1


def reverseTree(root):
	if not root:
		return
	root.left, root.right = root.right, root.left
	reverseTree(root.left)
	reverseTree(root.right)

class Node:
    def __init__(self, key) -> None:
        self.left = None
        self.right = None
        self.data = key

def inOrder(root):
	if not root:
		return
	inOrder(root.left)
	print(root.data, end=", ")
	inOrder(root.right)

root = Node(4)

# Left Sub Tree
root.left = Node(3)
root.left.left = Node(1)
root.left.right = Node(2)
# Right Sub tree
root.right = Node(6)
root.right.left = Node(5)
root.right.right = Node(7)

inOrder(root)
print()
reverseTree(root)
print()
inOrder(root)
