
import CreateLinkedList as LinkedList

def mergeLists(head1, head2):
    while(head1 is not None and head2 is not None):
        temp = head2.next
        head2.next = head1.next
        head1.next = head2
        head2 = temp
        head1 = head1.next.next

mergeLists(LinkedList.list1.head, LinkedList.list2.head)
print("\n")
LinkedList.print_list(LinkedList.list1)
