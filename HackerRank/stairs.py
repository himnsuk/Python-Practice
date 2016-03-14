#!/bin/python

import sys

n = int(raw_input().strip())
i = 1
while i <= n:
  p = 0
  stair = ""
  while p < n:
    if p >= n-i:
      stair = stair + "#"
    else:
      stair = stair + " "
    p = p + 1
  print stair
  i = i + 1

