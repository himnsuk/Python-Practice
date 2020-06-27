"""  replace with zero in row and colum if 0 found in matrix """


def print_mat(mat):
  r_len = len(mat)
  c_len = len(mat[0])

  for x in range(r_len):
    for y in range(c_len):
      print(mat[x][y], end=" ")
    print("")

def remove_zero_row_column_from_mat(mat):
  n_r = len(mat)
  n_c = len(mat[0])

  temp = []
  for i in range(n_r):
    for j in range(n_c):
      if mat[i][j] == 0:
        temp.append([i,j])
  
  for x in temp:
    print(x)
    for i in range(n_r):
      mat[i][x[1]] = 0
    for j in range(n_c):
      mat[x[0]][j] = 0 
  print_mat(mat)  
  return mat



matrix = [[1,2,3,4], [5,0,7,8], [9,10,0,12], [13,14,15,16]]
remove_zero_row_column_from_mat(matrix)
