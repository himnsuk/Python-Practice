
import CreateTree as Tree

def max(a,b):
    return a if a > b else b

def treeHeight(root):
    if root == None:
        return 0
    else:
        l_height = treeHeight(root.left)
        r_height = treeHeight(root.right)
        return max(l_height, r_height) + 1

print(treeHeight(Tree.root))
