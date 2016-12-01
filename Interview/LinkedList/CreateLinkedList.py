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

    def append_node(self, new_data):
        if self.head is None:
            temp = Node(new_data)
            self.head = temp
            return
        node = self.head
        while(node.next):
            node = node.next

        temp = Node(new_data)
        node.next = temp

    def reverseList(self):
        p = None
        c = self.head
        n = c.next
        while(n):
            c.next = p
            p = c
            c = n
            n = n.next
        c.next = p
        self.head = c

    def reverseGroup(self,head, k):
        count = 0
        c = head
        n = None
        p = None
        while(count < k and c is not None):
            n = c.next
            c.next = p
            p = c
            c = n
            count += 1

        if n is not None:
            head.next = self.reverseGroup(n, k)

        return p

def print_list(list):
    current = list.head
    while current != None:
        print(int(current.get_data()), end=" ")
        current = current.get_next()



mylist = UnorderedList()

for i in range(120, 10, -10):
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
letterList2 = UnorderedList()

letterList2.add_node('a')
letterList2.add_node('b')
letterList2.add_node('c')
letterList2.add_node('d')
letterList2.add_node('a')
letterList2.add_node('b')
letterList2.add_node('p')

list1 = UnorderedList()
list1.append_node(5)
list1.append_node(7)
list1.append_node(17)
list1.append_node(13)
list1.append_node(11)

list2 = UnorderedList()
list2.append_node(12)
list2.append_node(10)
list2.append_node(2)
list2.append_node(4)
list2.append_node(6)

# print_list(list1)
# print("\n")
#
# list2.reverseList()
# print_list(list2)
# mylist.head = mylist.reverseGroup(mylist.head, 3)
# print_list(mylist)
