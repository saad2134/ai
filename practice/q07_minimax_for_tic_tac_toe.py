# Minimax Algorithm on Tic-Tac-Toe

def print_board(b):
    for i in range(3):
        print(' | '.join(b[3*i:3*i+3]))
        if i<2: print('-'*9)
    print('-----------------')    

def winner(b):
    wins=[(0,1,2),(3,4,5),(6,7,8),(0,3,6),
          (1,4,7),(2,5,8),(0,4,8),(2,4,6)]
    for a,b2,c in wins:
        if b[a]==b[b2]==b[c] and b[a]!=' ': return b[a]
    if ' ' not in b: return 'Draw'
    return None

def minimax(b,player):
    w=winner(b)
    if w=='X': return 1,None
    if w=='O': return -1,None
    if w=='Draw': return 0,None
    if player=='X':
        best=-999;move=None
        for i in range(9):
            if b[i]==' ':
                b[i]='X'
                score,_=minimax(b,'O')
                b[i]=' '
                if score>best:best,move=score,i
        return best,move
    else:
        best=999;move=None
        for i in range(9):
            if b[i]==' ':
                b[i]='O'
                score,_=minimax(b,'X')
                b[i]=' '
                if score<best:best,move=score,i
        return best,move

def play():
    board=[' ']*9;current='X'
    print("You=O, AI=X")
    print('-----------------')    
    while True:
        print_board(board)
        w=winner(board)
        if w: print("The winner is:",w);return
        if current=='O':
            m=int(input("Your move 0-8:"))
            if board[m]==' ':
                board[m]='O';current='X'
        else:
            _,mv=minimax(board,'X')
            board[mv]='X';current='O'

if __name__=="__main__":
    play()
