import sys

t = int(raw_input().strip())
for x in xrange(t):
  n = int(raw_input().strip())
  a = int(raw_input().strip())
  b = int(raw_input().strip())

  def possible_stone_number(a, b, n):
    if a == b:
      return [a * (n -1)]
    if a < b:
      return possible_stone_number(b,a,n)
    return [ a * i + b * (n - i - 1) for i in xrange(n)]
  print  ' '.join(map(str,possible_stone_number(a, b, n)))


