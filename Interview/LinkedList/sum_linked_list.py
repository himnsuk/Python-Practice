from link_list import LinkedList, Node

def create_linked_list_from_number(num):
    ll = LinkedList()
    if num == 0:
        ll.add_node_in_end(Node(num))
        return ll
    while num > 0:
        a = num % 10
        ll.add_node_in_end(Node(a))
        num = num // 10
    return ll

def sum_linked_list(l1, l2):
    n1 = l1.head
    n2 = l2.head
    carry = 0
    ll3 = LinkedList()
    while n1 is not None and n2 is not None:
        s = n1.data + n2.data + carry
        if s >= 10:
            carry = s // 10
            s = s % 10
        else:
            carry = 0
        ll3.add_node_in_end(Node(s))
        n1 = n1.next
        n2 = n2.next

    while n1 is not None:
        s = n1.data + carry
        if s >= 10:
            carry = s // 10
            s = s % 10
        else:
            carry = 0
        ll3.add_node_in_end(Node(s))
        n1 = n1.next

    while n2 is not None:
        s = n2.data + carry
        if s >= 10:
            carry = s // 10
            s = s % 10
        else:
            carry = 0
        ll3.add_node_in_end(Node(s))
        n2 = n2.next
    print("")
    ll3.print_linked_list()
    return ll3


def create_number_from_linked_list(ll):
    n = ll.head
    total = 0
    i = 0
    while n is not None:
        total = total + n.data *(10**i)
        i += 1
        n = n.next 
    return total


if __name__ == "__main__":
    n1 = 8794
    n2 = 83594
    # n1 = 1
    # n2 = 1
    l1 = create_linked_list_from_number(n1)
    l2 = create_linked_list_from_number(n2)
    l1.print_linked_list()
    print("")
    l2.print_linked_list()
    l3 = sum_linked_list(l1, l2)
    x = n1 + n2
    num = create_number_from_linked_list(l3)
    print("\n", num)
    if x == num:
        print("Your code working awesome")
        