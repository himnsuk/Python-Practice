# inputs (3,7),(4,8),(1,2),(9,16),(13,18),(22,28)

import sys

st = input("enter intervals")

parse1 = st.replace("(", "")
parse2 = parse1.split("),")
parse2[len(parse2) - 1] = parse2[len(parse2) - 1].replace(")", "")

print(parse2)
