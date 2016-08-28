class Node:
    def __init__(self, data):
        self.data = data
        self.next = None


class LinkedList:
    def __init__(self):
        self.head = None


def create_linked_list():
    ll = LinkedList()
    n1 = Node(1)
    n2 = Node(2)
    n3 = Node(3)

    ll.head = n1
    n1.next = n2
    n2.next = n3
    print_linked_list(ll)


def print_linked_list(root):
    node = root.head
    while node != None:
        print(node.data, end=" -> ")
        node = node.next


def reverseLinkeList(head):
    first = head
    if not head:
        return head
    second = head.next

    first = None
    while second:
        temp, second.next = second.next, first
        first = temp
    head = first


if __name__ == "__main__":
    create_linked_list()
