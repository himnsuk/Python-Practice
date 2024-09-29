from multiprocessing import Process
import os
import time
def first_fn():
  i = 0
  while True:
    time.sleep(1)
    print(f"first function ====> {i}")
    i-=1

def second_fn():
  i = 0
  while True:
    time.sleep(2)
    print(f"Second function ====> {i}")
    i+=1
if __name__ == '__main__':
    # info('main line')
    p1 = Process(target=first_fn)
    p2 = Process(target=second_fn)
    p1.start()
    p2.start()