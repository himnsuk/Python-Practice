class Stack:
    def __init__(self):
        self.stack = []

    def push(self, data):
        self.stack.append(data)
        return self.stack
    
    def pop(self):
        return self.stack.pop()
    
    def isEmpty(self):
        if len(self.stack) == 0:
            return True
        return False
    
    def print_stack(self):
        while self.stack:
            print(self.stack.pop())
    
    def peek(self):
        if len(self.stack) == 0:
            print("Empty")
            return
        return self.stack[-1]


def sort_stack(st):
    s_st = Stack()
    print(st.isEmpty())
    if st.isEmpty():
        st.print_stack()
        return print("Stack is empty")
    
    s_st.push(st.pop())

    while not st.isEmpty():
        p1 = st.peek()
        p2 = s_st.peek()
        if p1 > p2:
            temp = st.pop()
            st.push(s_st.pop())
            while not s_st.isEmpty() and temp > s_st.peek():
                st.push(s_st.pop())
            s_st.push(temp)
            
        else:
            s_st.push(st.pop())
    return s_st

if __name__ == "__main__":
    st = Stack()

    st.push(5) 
    st.push(2) 
    st.push(4) 
    st.push(7) 
    st.push(9) 
    st.push(1) 
    y = sort_stack(st)
    y.print_stack()