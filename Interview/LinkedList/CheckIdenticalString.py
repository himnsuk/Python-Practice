
import CreateLinkedList as LinkedList

def stringCompare(head1, head2):
  if head1 is None and head2 is None:
    return 1
  else:
    while(head1 is not None and head2 is not None):
      if head1.data != head2.data:
        return -1
      else:
        head1 = head1.next
        head2 = head2.next
    return 1

print(stringCompare(LinkedList.letterList.head, LinkedList.letterList2.head))
