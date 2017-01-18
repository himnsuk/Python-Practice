#!/bin/bash

import sys
import pdb

class Node:

    def __init__(self,  data):
        self.data = data
        self.next = None

class LinkedList:

    def __init__(self):
        self.head = None

def CreateLinkedList(first, arr):

    for i in arr:
        temp = Node(i)
        temp.next = first.head
        first.head = temp
    return first

def PrintList(point):
    head = point.head
    while head:
        print(head.data)
        head = head.next


def sum_three_linked_list(l1, l2, l3):
    list1 = l1.head
    list2 = l2.head
    list3 = l3.head
    result_list = LinkedList()
    result = 0
    carry = 0
    while list1 != None and list2 != None and list3 != None:
        result = list1.data + list2.data + list3.data + carry
        if result >= 10:
            carry = int(result / 10)
            result = result % 10
        else:
            carry = 0
        temp = Node(result)
        temp.next = result_list.head
        result_list.head = temp
        list1 = list1.next
        list2 = list2.next
        list3 = list3.next

    while list1 == None and list2 != None and list3 != None:
        result = list2.data + list3.data + carry
        if result >= 10:
            carry = int(result / 10)
            result = result % 10
        else:
            carry = 0
        temp = Node(result)
        temp.next = result_list.head
        result_list.head = temp
        list2 = list2.next
        list3 = list3.next

    while list1 != None and list2 == None and list3 != None:
        result = list1.data + list3.data + carry
        if result >= 10:
            carry = int(result / 10)
            result = result % 10
        else:
            carry = 0
        temp = Node(result)
        temp.next = result_list.head
        result_list.head = temp
        list1 = list1.next
        list3 = list3.next

    while list1 != None and list2 != None and list3 == None:
        result = list1.data + list2.data + carry
        if result >= 10:
            carry = int(result / 10)
            result = result % 10
        else:
            carry = 0
        temp = Node(result)
        temp.next = result_list.head
        result_list.head = temp
        list1 = list1.next
        list2 = list2.next

    while list1 == None and list2 == None and list3 != None:
        result = list3.data + carry
        if result >= 10:
            carry = int(result / 10)
            result = result % 10
        else:
            carry = 0
        temp = Node(result)
        temp.next = result_list.head
        result_list.head = temp
        list3 = list3.next

    while list1 != None and list2 == None and list3 == None:
        result = list1.data + carry
        if result >= 10:
            carry = int(result / 10)
            result = result % 10
        else:
            carry = 0
        temp = Node(result)
        temp.next = result_list.head
        result_list.head = temp
        list1 = list1.next

    while list1 == None and list2 != None and list3 == None:
        result = list2.data + carry
        if result >= 10:
            carry = int(result / 10)
            result = result % 10
        else:
            carry = 0
        temp = Node(result)
        temp.next = result_list.head
        result_list.head = temp
        list1 = list1.next

    if carry > 0:
        temp = Node(carry)
        temp.next = result_list.head
        result_list.head = temp

    reverseList(result_list)

def reverseList(result_list):
    prev = result_list.head
    nex = result_list.head
    curr = result_list.head
    while curr != None:
        nex = curr.next
        curr.next = prev
        prev = curr
        curr = nex

    result_list.head = prev
    PrintList(result_list) 

n = int(input("Enter number of linked list"))
a =[]
for i in range(n):
    a.append(list(map(int, input().strip().split('->'))))

l1 =  LinkedList()
l2 =  LinkedList()
l3 =  LinkedList()

l1 = CreateLinkedList(l1, a[0])
l2 = CreateLinkedList(l2, a[1])
l3 = CreateLinkedList(l3, a[2])

sum_three_linked_list(l1, l2, l3)
