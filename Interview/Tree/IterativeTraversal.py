
import CreateTree as Tree

def InorderTraversal(root):
    stack = list()
    stack.append(root)
    node = root.left
    done = 0
    while(not done):
        if node != None:
            stack.append(node)
            node = node.left
        else:
            if(len(stack) > 0):
                node = stack.pop()
                print(node.data)
                node = node.right
            else:
                done = 1

def PreOrder(root):
    stack = list()
    node = root
    stack.append(node)
    done = 0
    while(not done):
        if len(stack) > 0:
            node = stack.pop()
            print(node.data)
            if node.right != None:
                stack.append(node.right)
            if node.left != None:
                stack.append(node.left)
        else:
            done = 1

def PostOrder(root):
    st1 = list()
    st2 = list()
    st1.append(root)
    while st1:
        node = st1.pop()
        st2.append(node)
        if node.left != None:
            st1.append(node.left)
        if node.right != None:
            st1.append(node.right)

    while st2:
        node = st2.pop()
        print(node.data)
# InorderTraversal(Tree.root)
# PreOrder(Tree.root)
PostOrder(Tree.root)
