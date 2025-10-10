import random

class TicTacToeBot:
    def __init__(self):
        self.board = [' ' for _ in range(9)]
        self.current_winner = None
    
    def print_board(self):
        for i in range(3):
            print(f' {self.board[i*3]} | {self.board[i*3+1]} | {self.board[i*3+2]} ')
            if i < 2:
                print('-----------')
    
    def available_moves(self):
        return [i for i, spot in enumerate(self.board) if spot == ' ']
    
    def make_move(self, square, letter):
        if self.board[square] == ' ':
            self.board[square] = letter
            if self.winner(square, letter):
                self.current_winner = letter
            return True
        return False
    
    def winner(self, square, letter):
        # Check row
        row_ind = square // 3
        row = self.board[row_ind*3:(row_ind+1)*3]
        if all([spot == letter for spot in row]):
            return True
        
        # Check column
        col_ind = square % 3
        column = [self.board[col_ind+i*3] for i in range(3)]
        if all([spot == letter for spot in column]):
            return True
        
        # Check diagonals
        if square % 2 == 0:
            diagonal1 = [self.board[i] for i in [0, 4, 8]]
            if all([spot == letter for spot in diagonal1]):
                return True
            diagonal2 = [self.board[i] for i in [2, 4, 6]]
            if all([spot == letter for spot in diagonal2]):
                return True
        return False
    
    def bot_move(self):
        # Simple AI: Try to win, then block, then random
        available = self.available_moves()
        
        # Check for winning move
        for move in available:
            self.board[move] = 'O'
            if self.winner(move, 'O'):
                return move
            self.board[move] = ' '
        
        # Check for blocking move
        for move in available:
            self.board[move] = 'X'
            if self.winner(move, 'X'):
                self.board[move] = ' '
                return move
            self.board[move] = ' '
        
        # Try center, then corners, then edges
        if 4 in available:
            return 4
        
        corners = [0, 2, 6, 8]
        empty_corners = [move for move in corners if move in available]
        if empty_corners:
            return random.choice(empty_corners)
        
        return random.choice(available)

def play_tic_tac_toe():
    game = TicTacToeBot()
    human_letter = 'X'
    bot_letter = 'O'
    
    print("=== Tic Tac Toe Bot ===")
    print("Positions:")
    print(" 0 | 1 | 2 ")
    print("-----------")
    print(" 3 | 4 | 5 ")
    print("-----------")
    print(" 6 | 7 | 8 ")
    print()
    
    while ' ' in game.board and not game.current_winner:
        game.print_board()
        
        # Human move
        try:
            move = int(input("\nYour move (0-8): "))
            if move not in game.available_moves():
                print("Invalid move! Try again.")
                continue
            game.make_move(move, human_letter)
        except ValueError:
            print("Please enter a number 0-8")
            continue
        
        if game.current_winner:
            break
            
        # Bot move
        if game.available_moves():
            bot_move = game.bot_move()
            print(f"Bot plays at position {bot_move}")
            game.make_move(bot_move, bot_letter)
    
    game.print_board()
    if game.current_winner:
        print(f"\n{game.current_winner} wins!")
    else:
        print("\nIt's a tie!")

if __name__ == "__main__":
    play_tic_tac_toe()