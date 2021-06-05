

cost = [[0]*21]*21
dp = [[0]*(1<<21)]*21

def solve(i, mask, n):
    if i == n:
        return 0
    
    if dp[i][mask] != -1:
        return dp[i][mask]
    
    