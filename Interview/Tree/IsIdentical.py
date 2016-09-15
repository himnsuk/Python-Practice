
import CreateTree as Tree

def isIdentical(root1, root2):
    if root1 is None and root2 is None:
        return True
    else:
        return (root1.data == root2.data and isIdentical(root1.left, root2.left) and isIdentical(root1.right, root2.right))
    

print(isIdentical(Tree.root, Tree.root2))
