def numIslands(grid):
    """
    :type grid: List[List[str]]
    :rtype: int
    """
    m = len(grid)
    n = len(grid[0])
    i, j, count_i = 0, 0, 0
    while(i < m):
        while(j < n):
            if(int(grid[i][j]) == 1):
                k = i
                l = j
                while(k < m and int(grid[k][j]) == 1):
                    k += 1
                while(l < n and int(grid[i][l]) == 1):
                    l += 1

                count_i += 1
                i = k
                j = l
            else:
                j += 1
        i += 1

    return count_i

grid = [["1","1","1","1","0"],["1","1","0","1","0"],["1","1","0","0","0"],["0","0","0","0","0"]]

print(numIslands(grid))
