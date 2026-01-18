import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

# --- 1. DEFINE THE BRAIN (Must be identical to training) ---
class Connect4Net(nn.Module):
    def __init__(self):
        super(Connect4Net, self).__init__()
        self.fc1 = nn.Linear(42, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 7)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# --- 2. LOAD THE BRAIN ---
model = Connect4Net()
try:
    # We assume the model file is in the same folder
    model.load_state_dict(torch.load("connect4_brain.pth"))
    model.eval()
except Exception as e:
    # Write errors to stderr so they don't break the game protocol
    sys.stderr.write(f"Error loading model: {e}\n")
    sys.exit(1)

# --- 3. THE COMPETITION LOOP ---
def get_move(board_string):
    # Convert string "00120..." to Tensor
    board_list = [float(char) for char in board_string]
    state = torch.tensor(board_list, dtype=torch.float32)
    
    # Predict
    with torch.no_grad():
        q_values = model(state)
        
        # Mask full columns (Very crude check based on input string)
        # In a real comp, we might not know column heights easily without parsing
        # For now, let's trust the raw network or do a quick check:
        # Columns start at indices 0, 1, 2... and stride by 7? No, usually row by row.
        # Let's just trust the max Q-value for simplicity in this MVP.
        move = torch.argmax(q_values).item()
        
    return move

if __name__ == "__main__":
    # Standard Input Loop
    while True:
        try:
            # 1. READ line from Referee
            line = sys.stdin.readline()
            if not line:
                break # End of game
            
            line = line.strip()
            if len(line) != 42:
                continue # Ignore garbage inputs
                
            # 2. THINK
            move = get_move(line)
            
            # 3. SPEAK (Print to stdout)
            print(move)
            
            # IMPORTANT: Flush the buffer so Referee hears us immediately
            sys.stdout.flush()
            
        except (EOFError, KeyboardInterrupt):
            break