import torch
import random
from connect4 import Connect4
from dqn_agent import Connect4Net, board_to_tensor

def human_agent(game):
    """Same human agent from before"""
    while True:
        try:
            user_input = input(f"Enter column (0-{game.cols-1}): ")
            col = int(user_input)
            if col < 0 or col >= game.cols:
                print("Invalid column.")
                continue
            if game.board[0][col] != 0:
                print("Column full.")
                continue
            return col
        except ValueError:
            print("Not a number.")

def play_game():
    # 1. LOAD THE BRAIN
    model = Connect4Net()
    try:
        model.load_state_dict(torch.load("connect4_brain.pth"))
        print("Brain loaded successfully!")
    except FileNotFoundError:
        print("Error: Could not find 'connect4_brain.pth'. Did you run training?")
        return

    model.eval() # Important: Switch to 'Evaluation Mode' (No learning, just predicting)

    # 2. SETUP GAME
    game = Connect4()
    print("--- AI (Player 1) vs HUMAN (Player 2) ---")
    game.print_board()
    
    game_over = False
    current_player = 1

    while not game_over:
        if 0 not in game.board[0]:
            print("Draw!")
            break

        if current_player == 1:
            # --- AI TURN ---
            print("AI is thinking...")
            state = board_to_tensor(game.board)
            with torch.no_grad(): # Don't calculate gradients (saves memory)
                q_values = model(state)
                
                # Filter out invalid moves (same as training)
                valid_cols = [c for c in range(game.cols) if game.board[0][c] == 0]
                for c in range(game.cols):
                    if c not in valid_cols:
                        q_values[c] = -9999
                
                # Pick the column with the highest score
                col = torch.argmax(q_values).item()
        else:
            # --- HUMAN TURN ---
            col = human_agent(game)

        # Drop Piece
        if game.drop_piece(col, current_player):
            # Check Win
            if game.check_winner(current_player):
                game.print_board()
                winner = "AI" if current_player == 1 else "HUMAN"
                print(f"!!! {winner} WINS !!!")
                game_over = True
            
            # Switch Turn
            current_player = 2 if current_player == 1 else 1
            
            if not game_over:
                game.print_board()

if __name__ == "__main__":
    play_game()