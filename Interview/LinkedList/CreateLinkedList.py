#!/bin/bash

import sys

class Node:
    def __init__(self, data):
        self.data = data
        self.next = None

    def get_data(self):
        return self.data

    def set_data(self, new_data):
        self.data = new_data

    def get_next(self):
        return self.next

    def set_next(self, new_node):
        self.next = new_node

    def is_empty(self):
        self.next = None



class UnorderedList:
    def __init__(self):
        self.head = None

    def add_node(self, new_data):
        temp = Node(new_data)
        temp.set_next(self.head)
        self.head = temp

def print_list(list):
    current = list.head
    while current != None:
        print(int(current.get_data()), end=" ")
        current = current.get_next()



mylist = UnorderedList()

for i in range(10, 120, 10):
    mylist.add_node(i)
# print_list(mylist)
# print("\n")

letterList = UnorderedList()

letterList.add_node('a')
letterList.add_node('b')
letterList.add_node('c')
letterList.add_node('d')
letterList.add_node('a')
letterList.add_node('b')
letterList.add_node('c')
# print_list(letterList)
