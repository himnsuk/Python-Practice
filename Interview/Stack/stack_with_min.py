class Stack:
    min_val = 999999

    def __init__(self):
        self.stack = []
    def push(self, data):
        if data < self.min_val:
            self.min_val = data
        self.stack.append(data)
        return self.stack
    
    def pop(self):
        return self.stack.pop()
    
    def isEmpty(self):
        if len(self.stack == 0):
            return True
        return False

if __name__ == "__main__":
    st = Stack()

    st.push(5) 
    st.push(3) 

    print(st.min_val)