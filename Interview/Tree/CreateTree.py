#!/bin/bash

import sys

class Node:
    def __init__(self, key):
        self.left = None
        self.right = None
        self.data = key

root = Node(1)
root.left = Node(2)
root.right = Node(3)
root.left.left = Node(4)
root.right.right = Node(5)
root.left.left.left = Node(7)

root2 = Node(1)
root2.left = Node(2)
root2.right = Node(3)
root2.left.left = Node(4)
root2.right.right = Node(5)
root2.left.left.left = Node(8)
