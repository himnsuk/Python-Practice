#!/bin/bash

import sys
import CreateLinkedList as LinkedList

list1 = LinkedList.UnorderedList()
list2 = LinkedList.UnorderedList()

for i in range(1, 10, 2):
    list1.add_node(i)

for i in range(2, 10, 2):
    list2.add_node(i)

LinkedList.print_list(list1)
print("\n")
LinkedList.print_list(list2)

def addLists(list1, list2):
    list3 = LinkedList.UnorderedList()
    first = list1.head
    second = list2.head
    carry = 0
    while first != None and second != None:
        if carry != 0:
            sum = first.data + second.data + carry
            if sum > 10:
                carry = sum / 10
                sum = sum % 10
            else:
                carry = 0
        else:
            sum = first.data + second.data
            if sum > 10:
                carry = sum / 10
                sum = sum % 10
            else:
                carry = 0
        list3.add_node(sum)
        first = first.next
        second = second.next

    while first != None:
        if carry != 0:
            sum = first.data + carry
            if sum > 10:
                carry = sum / 10
                sum = sum % 10
            else:
                carry = 0
        else:
            sum = first.data 
            if sum > 10:
                carry = sum / 10
                sum = sum % 10
            else:
                carry = 0
        list3.add_node(sum)
        first = first.next

    while second != None:
        if carry != 0:
            sum = second.data + carry
            if sum > 10:
                carry = sum / 10
                sum = sum % 10
            else:
                carry = 0
        else:
            sum = second.data 
            if sum > 10:
                carry = sum / 10
                sum = sum % 10
            else:
                carry = 0
        list3.add_node(sum)
        second = second.next
    print("\n")
    LinkedList.print_list(list3)

addLists(list1, list2)
