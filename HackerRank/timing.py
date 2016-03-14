#!/bin/python

import sys

time = raw_input().strip()
timeCheck = map(int,time[:-2].strip().split(':'))
if timeCheck[0] <= 12 and timeCheck[1] < 60 and timeCheck[2] < 60:
  hour = 0
  time2 = []
  time24 = ''
  check = time[-2:]

  if check == 'PM' and timeCheck[0] != 12:
    hour = int(time.strip().split(':')[0]) + 12
    time2 = time[:-2].strip().split(':')
    time2[0] = str(hour)
    print time2
    time24 = ':'.join(time2)
    print time24

  elif check == 'AM' and timeCheck[0] == 12:
    time2 = time[:-2].strip().split(':')
    time2[0] = '00'
    print ':'.join(time2)
  else:
    print time[:-2]
