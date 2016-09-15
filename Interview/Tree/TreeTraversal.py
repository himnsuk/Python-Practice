
import CreateTree as Tree

def printInOrder(root):
    if root:
        printInOrder(root.left)
        print(root.data, end=" ")
        printInOrder(root.right)
        
def printPreOrder(root):
    if root:
        print(root.data, end=" ")
        printPreOrder(root.left)
        printPreOrder(root.right)
        
def printPostOrder(root):
    if root:
        printPostOrder(root.left)
        printPostOrder(root.right)
        print(root.data, end=" ")

print("Inorder Traversal")
printInOrder(Tree.root)
print("\n Preorder Traversal \n")
printPreOrder(Tree.root)
print("\n Postorder Traversal \n")
printPostOrder(Tree.root)
