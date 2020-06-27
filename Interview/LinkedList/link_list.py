class Node:
    def __init__(self, data):
        self.data = data
        self.next = None


class LinkedList:
    def __init__(self):
        self.head = None

    def add_node_in_end(self, node):
        if self.head is None:
            self.head = node
        else:
            n = self.head
            while n.next is not None:
                n = n.next
            n.next = node

    def add_in_begining(self, node):
        if self.head is None:
            self.head = node
        else:
            node.next = self.head
            self.head = node

    def delete_end(self):
        if self.head is None:
            print("Linked List is empty")
        else:
            n = self.head
            while n.next.next is not None:
                n = n.next

            n.next = None

    def delete_begining(self):
        if self.head is None:
            print("Linked List is empty")
        else:
            n = self.head
            self.head = n.next
            n.next = None

    def print_linked_list(self):
        n = self.head
        print("(head) ", end = "-> ")
        while n is not None:
            print(n.data, end=" -> ")
            n = n.next
        print("None")

        while n is not None:
            print(n.data, end=" -> ")
            n = n.next


if __name__ == "__main__":
    ll = LinkedList()
    for x in range(10):
        node = Node(x)
        ll.add_node_in_end(node)

    print("Print Created List")
    ll.print_linked_list()

    print("\n Add node in the start of linked list")
    n10 = Node(10)
    ll.add_in_begining(n10)

    ll.print_linked_list()

    print("\n Delete item from beginning")
    ll.delete_begining()
    ll.print_linked_list()

    print("\n Delete node from the end")
    ll.delete_end()
    ll.print_linked_list()
