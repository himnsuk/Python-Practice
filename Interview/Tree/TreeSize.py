
import CreateTree as Tree

def treeSize(root):
    if root == None:
        return 0
    else:
        return (treeSize(root.left) + 1 + treeSize(root.right))


print(treeSize(Tree.root))
