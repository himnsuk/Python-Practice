class A(object):
    def __init__(self, i=0):
        self.i = i
    def m1(self):
        self.i += 1
class B(A):
    def __init__(self, j=0):
        A.__init__(self, 3)
        self.j = j
    def m1(self):
        self.j += 1
def main():
    b = B()
    b.m1()
    print(b.i, b.j)
main()
