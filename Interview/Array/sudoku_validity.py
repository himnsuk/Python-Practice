import math as m

board = [[7, 9, 2, 1, 5, 4, 3, 8, 6],
         [6, 4, 3, 8, 2, 7, 1, 5, 9],
         [8, 5, 1, 3, 9, 6, 7, 2, 4],
         [2, 6, 5, 9, 7, 3, 8, 4, 1],
         [4, 8, 9, 5, 6, 1, 2, 7, 3],
         [3, 1, 7, 4, 8, 2, 9, 6, 5],
         [1, 3, 6, 7, 4, 8, 5, 9, 2],
         [9, 7, 4, 2, 1, 5, 6, 3, 8],
         [5, 2, 8, 6, 3, 9, 4, 1, 7]]



def has_duplicate(block):
    block = list(filter(lambda x: x != 0, block))
    return len(block) != len(set(block))

def isValidSudoku(board):
    n = len(board)

    if any(
        has_duplicate([board[i][j] for j in range(n)])
        or
        has_duplicate([board[j][i] for j in range(n)])
        for i in range (n)
    ):
        return False

    grid_size = int(m.sqrt(n))
    
    return all(
        not has_duplicate([board[a][b]
        for a in range(grid_size*I, grid_size*(I+1))
        for b in range(grid_size*J, grid_size*(J+1))
        ]) for I in range(grid_size) for J in range(grid_size)
    )

print(isValidSudoku(board))