#!/bin/bash

import sys
from collections import deque

queue = deque(["Eric", "John", "Michael"])
queue.append('Terry')
print(queue)
queue.append('Graham')
print(queue)

print(queue.popleft())

print(queue)

