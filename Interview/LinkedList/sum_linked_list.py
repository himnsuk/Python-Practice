from link_list import LinkedList, Node

def create_linked_list_from_number(num):
    ll = LinkedList()
    ll_len = 0
    while num >= 0:
        a = num % 10
        ll.add_node_in_end(Node(a))
        num = num // 10
        ll_len += 1
    return (ll, ll_len)

def sum_linked_list(l1, l2):
    n1 = l1.head
    n2 = l2.head
    