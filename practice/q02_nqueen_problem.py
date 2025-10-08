N = 5
found=False
def is_safe(matrix,row,col):
    for i in range (row):
        if matrix[i] ==  col:
            return False
        if abs(matrix [i]-col) == abs(i-row):
            return False
    return True
def dfs(matrix,row):
    global found
    if found:
        return
    if row == N:
        print_solution(matrix)
        found=True
        return
    for col in range(N):
        if is_safe(matrix,row,col):
            matrix[row]=col
            dfs(matrix,row+1)
            matrix[row]=-1
def print_solution(matrix):
    for r in range (N):
        print(" ".join(" Q " if matrix[r]==c else " . " for c in range(N)))
        print()
matrix = [-1]*N
dfs(matrix,0)

#------------------
# R N B K Q B N R |
# ♟♟♟♟♟ ♟♟♟ |
#                 |
#                 |
#                 |
#                 |
# ♟♟♟♟♟ ♟♟♟ |
# R N B Q K B N R |
#------------------

