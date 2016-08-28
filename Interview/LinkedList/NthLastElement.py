#!/bin/bash

import sys
import CreateLinkedList as LinkedList

def nthElement(list, n):
    current = list.head
    nth_element = list.head

    for i in range(n):
        current = current.next

    while current != None:
        current = current.next
        nth_element = nth_element.next
    return nth_element.data

print(nthElement(LinkedList.mylist, 2))

def middleElement(list):
    fast = slow = list.head
    while fast != None and fast.next != None:
        fast = fast.next.next
        slow = slow.next

    return slow.data

print(middleElement(LinkedList.mylist))


def removeInBetween(ptr):
    ptr_next = ptr.next
    ptr.data = ptr_next.data
    ptr.next = ptr_next.next
    ptr_next = None

removeInBetween(LinkedList.mylist.head.next.next.next.next)
LinkedList.print_list(LinkedList.mylist)
