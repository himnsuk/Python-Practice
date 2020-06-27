from link_list import Node, LinkedList


def remove_duplicate_with_buffer(ll):
    if ll.head is None:
        return
    j = ll.head
    k = j.next

    n_dict = {j.data: 1}
    while k is not None:
        if k.data in n_dict:
            j.next = k.next
            k.next = None
            k = j.next
        else:
            n_dict[k.data] = 1
            print(n_dict)
            j = j.next
            k = k.next
        print("")
        # ll.print_linked_list()

def remove_duplicate_without_buffer(ll):
    if ll.head is None:
        return
    
    j = ll.head

    while j.next is not None:
        p = j
        k = j.next
        while k is not None:
            if k.data == j.data:
                p.next = k.next
                k.next = None
                k = p.next
            else:
                p = p.next
                k = k.next
        if j.next is not None:
            j = j.next
        
# arr = [1,3,3,4,4,6,6,5]
# arr = [1, 3, 3, 3, 3, 3, 3]

arr = [2,2,3,4]
# arr = []

if __name__ == "__main__":
    ll = LinkedList()
    for x in arr:
        node = Node(x)
        ll.add_node_in_end(node)

    # ll.print_linked_list()

    # remove_duplicate_with_buffer(ll)
    remove_duplicate_without_buffer(ll)
    print("")
    ll.print_linked_list()
