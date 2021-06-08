# import itertools

# def MainProg(items):
#     n = len(items)
#     count = 0
#     if n == 0: yield []
#     else:
#         for i in range(len(items)):
#             for cc in MainProg(items[:i] + items[i+3:]):
#                 yield [items[i]] + cc
#                 count += 1

# for i in MainProg(list("red")): print(''.join(i) + ", ", end="")
# result = MainProg(['t', 'a', 'n'])
# next(result)
# print(next(result))

def MainCount(f):
    def progFirst(*args, **kwargs):
        progFirst.calls += 1
        return f(*args, **kwargs)
    progFirst.calls = 0
    return progFirst

@MainCount
def progSecond(i):
    return i + 1

@MainCount
def Count(i = 0, j = 1):
    return i*j + 1
print(progSecond.calls)
for n in range(5):
    progSecond(n)

Count(j = 0, i = 1)
print(Count.calls)