import random

def evaluate_window(window, piece):
    score = 0
    opp_piece = 1
    if piece == 1: opp_piece = 2
    
    # 1. PRIORITY: Connect 4 (Winning)
    if window.count(piece) == 4:
        score += 100
    
    # 2. Strong Position: 3 pieces + 1 empty
    elif window.count(piece) == 3 and window.count(0) == 1:
        score += 5
        
    # 3. Decent Position: 2 pieces + 2 empty
    elif window.count(piece) == 2 and window.count(0) == 2:
        score += 2
        
    # 4. DEFENSE: Block opponent's 3-in-a-row
    # We punish the board score heavily if the opponent has 3 in a row
    if window.count(opp_piece) == 3 and window.count(0) == 1:
        score -= 4 # Negative score means "Bad board for us"
        
    return score

def score_position(game, piece):
    score = 0
    
    # A. CENTER COLUMN PREFERENCE
    # We like pieces in the center column (index 3) because it enables more wins
    center_array = [row[3] for row in game.board]
    center_count = center_array.count(piece)
    score += center_count * 3

    # B. SCAN BOARD (Horizontal, Vertical, Diagonal)
    # This is similar to check_winner, but we sum up scores instead of returning True/False
    
    # Horizontal
    for r in range(game.rows):
        for c in range(game.cols - 3):
            window = [game.board[r][c+i] for i in range(4)]
            score += evaluate_window(window, piece)

    # Vertical
    for r in range(game.rows - 3):
        for c in range(game.cols):
            window = [game.board[r+i][c] for i in range(4)]
            score += evaluate_window(window, piece)

    # Positive Diagonal
    for r in range(3, game.rows):
        for c in range(game.cols - 3):
            window = [game.board[r-i][c+i] for i in range(4)]
            score += evaluate_window(window, piece)
            
    # Negative Diagonal
    for r in range(game.rows - 3):
        for c in range(game.cols - 3):
            window = [game.board[r+i][c+i] for i in range(4)]
            score += evaluate_window(window, piece)

    return score

# To simplify, we need a valid_locations helper
def get_valid_locations(game):
    return [c for c in range(game.cols) if game.board[0][c] == 0]

def minimax(game, depth, maximizingPlayer, player_piece):
    valid_locations = get_valid_locations(game)
    is_terminal = game.check_winner(1) or game.check_winner(2) or len(valid_locations) == 0
    
    # 1. BASE CASE: Stop recursion
    if depth == 0 or is_terminal:
        if is_terminal:
            if game.check_winner(player_piece):
                return (None, 10000000) # We won!
            elif game.check_winner(3 - player_piece):
                return (None, -10000000) # We lost!
            else: 
                return (None, 0) # Draw
        else: # Depth is 0
            return (None, score_position(game, player_piece))

    # 2. MAXIMIZING STEP (The AI's Turn)
    if maximizingPlayer:
        value = -float('inf')
        column = random.choice(valid_locations)
        for col in valid_locations:
            # Simulate Move
            row = -1
            for r in range(game.rows -1, -1, -1):
                if game.board[r][col] == 0:
                    row = r
                    break
            game.board[row][col] = player_piece # Drop piece
            
            # RECURSE: Call minimax for the opponent (False)
            new_score = minimax(game, depth-1, False, player_piece)[1]
            
            # Undo Move
            game.board[row][col] = 0 
            
            if new_score > value:
                value = new_score
                column = col
        return column, value

    # 3. MINIMIZING STEP (The Opponent's Turn)
    else: 
        value = float('inf')
        column = random.choice(valid_locations)
        opponent_piece = 3 - player_piece
        for col in valid_locations:
            # Simulate Move
            row = -1
            for r in range(game.rows -1, -1, -1):
                if game.board[r][col] == 0:
                    row = r
                    break
            game.board[row][col] = opponent_piece # Drop Opponent Piece
            
            # RECURSE: Call minimax for the AI (True)
            new_score = minimax(game, depth-1, True, player_piece)[1]
            
            # Undo Move
            game.board[row][col] = 0 
            
            if new_score < value:
                value = new_score
                column = col
        return column, value
    
def minimax_agent(game, player):
    # Depth 4 means looking 4 moves ahead (My move, Your move, My move, Your move)
    # Warning: Depth 5 or 6 will be very slow in Python!
    col, minimax_score = minimax(game, 4, True, player)
    return col

def random_agent(game):
    """
    Input: The game object (so we can see the board)
    Output: A random valid column index (0-6)
    """
    valid_cols = [c for c in range(game.cols) if game.board[0][c] == 0]
    return random.choice(valid_cols)

def smart_agent(game, player):
    """
    1. Search for a winning move for 'player'.
    2. If none, search for a winning move for the 'opponent' (and block it).
    3. If neither, pick random.
    """
    valid_cols = [c for c in range(game.cols) if game.board[0][c] == 0]
    
    # Identify who the opponent is
    # If player is 1, opponent is 2. If player is 2, opponent is 1.
    opponent = 3 - player 
    
    # --- STEP 1: ATTACK (Check if WE can win) ---
    for col in valid_cols:
        # Find the row where the piece lands
        target_row = -1
        for row in range(game.rows -1, -1, -1):
            if game.board[row][col] == 0:
                target_row = row
                break
        
        # Simulate OUR move
        game.board[target_row][col] = player
        if game.check_winner(player):
            game.board[target_row][col] = 0 # Undo
            return col # TAKE THE WIN!
        game.board[target_row][col] = 0 # Undo

    # --- STEP 2: DEFENSE (Check if THEY can win) ---
    for col in valid_cols:
        # Find the row where the piece lands
        target_row = -1
        for row in range(game.rows -1, -1, -1):
            if game.board[row][col] == 0:
                target_row = row
                break
                
        # Simulate THE OPPONENT'S move
        game.board[target_row][col] = opponent
        if game.check_winner(opponent):
            # If they win here, we MUST play here to block them
            game.board[target_row][col] = 0 # Undo
            print(f"Blocking opponent in column {col}") # Optional: Let us know it blocked
            return col 
        game.board[target_row][col] = 0 # Undo

    # --- STEP 3: RANDOM ---
    return random.choice(valid_cols)

def human_agent(game):
    """
    Input: The game object (so we can see the board)
    Output: A column index (0-6) chosen by the human player
    """
    while True:
        try:
            user_input = input(f"Enter a column (0-{game.cols-1}): ")
            col = int(user_input)

            if col < 0 or col >= game.cols:
                print("Invalid column! Please pick a number between 0 and 6.")
                continue

            if game.board[0][col] != 0:
                print("That column is full! Pick another one.")
                continue

            return col
        
        except ValueError:
            print("That's not a number! Please enter an integer.")

class Connect4:
    def __init__(self):
        self.rows = 6
        self.cols = 7

        self.board = [[0 for _ in range(self.cols)] for _ in range(self.rows)]

    def print_board(self):
        print("\n  0 1 2 3 4 5 6 (column numbers)")
        print(" ---------------")
        for row in self.board:
            print("|", *row, "|")
        print(" ---------------")

    def drop_piece(self, col, player):
        if col < 0 or col >= self.cols:
            print(f"Error: Column {col} does not exist!")
            return False
        
        for row in range(self.rows -1, -1, -1):
            if self.board[row][col] == 0:
                self.board[row][col] = player
                return True
            
        print(f"Column {col} is full!")
        return False
    
    def check_winner(self, player):
        """
        Checks if the given 'player' (1 or 2) has won.
        Returns True if they won, False otherwise.
        """

        # Check horizontal
        for r in range(self.rows):
            for c in range(self.cols - 3):
                if all(self.board[r][c + i] == player for i in range(4)):
                    return True

        # Check vertical
        for r in range(self.rows - 3):
            for c in range(self.cols):
                if all(self.board[r + i][c] == player for i in range(4)):
                    return True
                
        # Check Positive Diagonal ( / )
        for r in range(3, self.rows):
            for c in range(self.cols - 3):
                if all(self.board[r - i][c + i] == player for i in range(4)):
                    return True
                
        # Check Negative Diagonal ( \ )
        for r in range(self.rows - 3):
            for c in range(self.cols - 3):
                if all (self.board[r + i][c + i] == player for i in range(4)):
                    print(f"Player {player} wins!")
                    return True
                
        return False
    
if __name__ == "__main__":
    game = Connect4()
    
    print("--- Starting Game: You (P1) vs Random (P2) ---")
    game.print_board()
    
    game_over = False
    current_player = 1
    
    while not game_over:
        if 0 not in game.board[0]:
            print("It's a Draw! Board is full.")
            break

        # ... inside the loop ...
        if current_player == 1:
            # Player 1 is SMART
            # Note: We have to pass 'current_player' so it knows who it is!
            col = smart_agent(game, current_player)
        else:
            # Player 2 is RANDOM
            col = human_agent(game)
        print(f"Player {current_player} drops in column {col}")
        
        # 2. Attempt to drop the piece
        success = game.drop_piece(col, current_player)
        
        if success:
            # 3. Check for a win
            if game.check_winner(current_player):
                game.print_board()
                print(f"!!! PLAYER {current_player} WINS !!!")
                game_over = True
            
            # 4. Switch turns
            # IMPORTANT: This block is aligned with 'if game.check_winner', NOT inside it.
            if current_player == 1:
                current_player = 2
            else:
                current_player = 1
            
            # Print board to see the battle happen step-by-step
            game.print_board()
            
        else:
            print("Invalid move, trying again...")

    print("Game Over.")