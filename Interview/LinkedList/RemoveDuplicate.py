#!/bin/bash

import sys
import CreateLinkedList as LinkedList

def removeDuplicate(list):
    dict = {}
    current = list.head
    if current.next == None:
        return

    dict[current.data] = 1
    previous = current
    current = current.next
    while current != None:
        if current.data in dict:
            previous.next = current.next
            current = current.next
        else:
            dict[current.data] = 1
            previous = current
            current = current.next

def removeDuplicateWithoutBuffer(list):
    current = list.head
    while current != None:
        inner = current.next
        previous = current
        while inner != None and current.next != None:
            if inner.data == current.data:
                previous.next = inner.next
                inner = inner.next
            else:
                previous = inner
                inner = inner.next
        current = current.next

removeDuplicateWithoutBuffer(LinkedList.letterList)
print("\n")
LinkedList.print_list(LinkedList.letterList)

