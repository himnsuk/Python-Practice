#!/bin/bash

import sys
import CreateLinkedList as LinkedList

# print(LinkedList.mylist.head.data)
LinkedList.print_list(LinkedList.mylist)
print("\n")

def delFirst(list):
    current = list.head
    list.head = current.next
    current.next = None

def search(list, item):
    current = list.head
    found = False
    while current != None:
        if current.data == item:
            found = True
            return found
        else:
            current = current.next
    return found

def findAndDelete(list, item):
    current = list.head
    previous = None
    while current != None:
        if current.data == item:
            if previous == None:
                list.head = current.next
            else:
                previous.next = current.next
            current.next = None
            return
        else:
            previous = current
            current = current.next

delFirst(LinkedList.mylist)
LinkedList.print_list(LinkedList.mylist)

# print(search(LinkedList.mylist, 20))

print("\n")
findAndDelete(LinkedList.mylist, 80)
LinkedList.print_list(LinkedList.mylist)
