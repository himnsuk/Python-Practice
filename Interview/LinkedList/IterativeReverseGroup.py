
import CreateLinkedList as LinkedList

def iterativeReverseGroup(head, k):
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
        head.next = reverseGroup(n, k)

    return p

LinkedList.print_list(LinkedList.mylist)
print("\n")
LinkedList.mylist.head = reverseGroup(LinkedList.mylist.head, 3)
LinkedList.print_list(LinkedList.mylist)