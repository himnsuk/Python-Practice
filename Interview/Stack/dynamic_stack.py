
class DynamicStack:
    def __init__(self, threshold):
        self.stack = []
        self.threshold = threshold
    
    def push(self, data):
        if not self.stack:
            l = [data]
            self.stack.append(l)
        else:
            last_stack = self.stack[-1]
            if len(last_stack) == self.threshold:
                l2 = [data]
                self.stack.append(l2)
            else:
                last_stack.append(data)

    def pop(self):
        if not self.stack:
            print("Stacke is empty")
            return 
        else:
            l = self.stack[-1]
            if not l:
                self.stack.pop()
                if self.stack:
                    l2 = self.stack[-1]
                    return l2.pop()
                else:
                    print("Stack is empty")
                    return
            else:
                return l.pop()

if __name__ == "__main__":
    st = DynamicStack(10)
    st.push(10)
    st.push(20)
    st.push(21)
    st.push(22)
    st.push(23)
    st.push(24)
    st.push(25)
    st.push(26)
    st.push(27)
    st.push(28)
    st.push(29)
    st.push(30)
    st.push(31)
    st.push(32)
    st.push(33)
    st.push(34)
    st.push(35)
    st.push(36)
    st.push(37)
    st.push(38)
    st.push(39)
    st.push(40)
    st.push(41)
    st.push(42)
    st.push(43)
    st.push(44)
    st.push(45)
    st.push(46)
    st.push(47)
    st.push(48)
    st.push(49)
    st.push(50)
    st.push(51)
    st.push(52)
    st.push(53)
    st.push(54)
    st.push(55)
    st.push(56)
    st.push(57)
    st.push(58)
    st.push(59)
    st.push(60)
    print(st.pop())

