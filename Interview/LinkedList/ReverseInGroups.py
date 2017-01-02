
import CreateLinkedList as LinkedList

def reverseGroup(head, k):
    c = head
    n = None
    p = None
    count = 0
    while(c is not None and count < k):
        n = c.next
        c.next = p
        p = c
        c = n
        count += 1
    if n is not None:
        head.next = reverseGroup(n, k)

    return p

LinkedList.print_list(LinkedList.mylist)

print("\n")

reverseGroup(LinkedList.mylist.head, 3)

LinkedList.print_list(LinkedList.mylist)
