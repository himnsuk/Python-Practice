#!/bin/python
import sys
n = int(raw_input().strip())
s = raw_input().strip()
k = int(raw_input().strip())

encrypted = ''
for character in s:
  if (ord(character) >= 65 and ord(character) <=90) or (ord(character) >= 97 and ord(character) <=122):
    char_nuumber = ord(character) + k
    while (char_nuumber > 90 and ord(character) >= 65 and ord(character) <=90):
      char_nuumber -= 26
    while (char_nuumber > 122 and ord(character) >= 97 and ord(character) <=122):
      char_nuumber -= 26
    character = chr(char_nuumber)
  encrypted += character
    
print encrypted

