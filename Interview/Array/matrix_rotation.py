""" Matrix Rotation Code Cockwise AntiClock Wise """
def print_mat(mat):
  r_len = len(mat)
  c_len = len(mat[0])

  for x in range(r_len):
    for y in range(c_len):
      print(mat[x][y], end=" ")
    print()
# Matrix rotation clockwise
def mat_rotate_clock(mat):
  r_len = len(mat)
  c_len = len(mat[0])

  for x in range(r_len//2):
    for y in range(x, c_len-x-1):
      temp = mat[x][y]

      mat[x][y] = mat[c_len-y-1][x]
      mat[c_len-y-1][x] = mat[c_len-x-1][c_len-y-1]
      mat[c_len-x-1][c_len-y-1] = mat[y][c_len-x-1]
      mat[y][c_len-x-1] = temp
  print_mat(mat)

# Matrix rotation anticlock wise
def mat_rotate_anti_clock(mat):
  n = len(mat[0])

  for i in range(n//2):
    for j in range(i, n-i-1):
      temp = mat[i][j]

      mat[i][j] = mat[j][n-1-i]
      mat[j][n-1-i] = mat[n-i-1][n-j-1]
      mat[n-i-1][n-j-1] = mat[n-j-1][i]
      mat[n-j-1][i] = temp

  print_mat(mat)


matrix = [[1,2,3,4], [5,6,7,8], [9,10,11,12], [13,14,15,16]]

print_mat(matrix)
print("Clockwise Rotation")
print("_________________")

mat_rotate_clock(matrix)

print("AntiClockwise Rotation")
print("_________________")
mat_rotate_anti_clock(matrix)
