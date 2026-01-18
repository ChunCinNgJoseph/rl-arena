import subprocess
import sys
from connect4 import Connect4

def run_match():
    # 1. Start the Agent Process
    # We use 'python -u' to ensure unbuffered output
    agent_process = subprocess.Popen(
        ['python', '-u', 'submission.py'],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=sys.stderr, # Let agent errors show up in our terminal
        text=True # Treat input/output as text strings, not bytes
    )

    game = Connect4()
    print("--- REFEREE: Starting Match (Human vs Agent) ---")
    game.print_board()
    
    game_over = False
    # Let's say Human is Player 1, Agent is Player 2
    current_player = 1 
    
    try:
        while not game_over:
            if 0 not in game.board[0]:
                print("Draw!")
                break

            if current_player == 1:
                # --- HUMAN TURN ---
                try:
                    col = int(input("Your Move (0-6): "))
                except ValueError:
                    continue
            else:
                # --- AGENT TURN ---
                # 1. Convert board to string protocol (000102...)
                board_str = ""
                for row in game.board:
                    for cell in row:
                        board_str += str(cell)
                
                # 2. Send to Agent
                agent_process.stdin.write(board_str + "\n")
                agent_process.stdin.flush()
                
                # 3. Read Agent's Move
                move_str = agent_process.stdout.readline().strip()
                if not move_str:
                    print("Agent crashed or sent nothing!")
                    break
                col = int(move_str)
                print(f"Agent chose column: {col}")

            # Execute Move
            if game.drop_piece(col, current_player):
                game.print_board()
                if game.check_winner(current_player):
                    winner = "Human" if current_player == 1 else "Agent"
                    print(f"!!! {winner} WINS !!!")
                    game_over = True
                
                current_player = 3 - current_player # Switch 1 <-> 2
            else:
                print("Invalid move detected.")

    except Exception as e:
        print(f"Match Error: {e}")
    finally:
        # Kill the agent process when game ends
        agent_process.terminate()

if __name__ == "__main__":
    run_match()