import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np
from connect4 import Connect4 # Import your game logic
from collections import deque

# --- 1. THE BRAIN ---
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

# --- 2. HELPER: BOARD TO TENSOR ---
# The neural net needs a flat list of 42 numbers, not a 6x7 grid.
def board_to_tensor(board):
    # Flatten the list of lists into one big list
    flat_board = []
    for row in board:
        flat_board.extend(row)
    # Convert to PyTorch Tensor
    return torch.tensor(flat_board, dtype=torch.float32)

# --- 3. THE SILENT GAME ---
# We inherit from your original class but disable the printing
class SilentConnect4(Connect4):
    def print_board(self):
        pass # Do nothing!

class ReplayBuffer:
    def __init__(self, capacity):
        # A deque is a list that automatically pops the oldest item 
        # when you add a new one if it's full.
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        # Save the experience tuple
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        # Pick 'batch_size' random memories
        # zip(*...) is a cool Python trick to unzip the list of tuples
        # It turns [(s1, a1, r1...), (s2, a2, r2...)] into ([s1, s2], [a1, a2]...)
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

# --- 4. THE TRAINING LOOP ---
def train_dqn():
    brain = Connect4Net()
    optimizer = optim.Adam(brain.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()
    
    # --- NEW: MEMORY ---
    memory = ReplayBuffer(10000) # Remember the last 10,000 moves
    batch_size = 64 # Learn from 64 moves at a time
    
    episodes = 2000 # Let's double the practice
    epsilon = 1.0
    decay = 0.995
    min_epsilon = 0.05 # Let it explore a bit less eventually
    
    print(f"Training with Replay Buffer for {episodes} games...")

    for episode in range(episodes):
        game = SilentConnect4()
        state = board_to_tensor(game.board)
        game_over = False
        
        while not game_over:
            # 1. Action (Same as before)
            valid_cols = [c for c in range(game.cols) if game.board[0][c] == 0]
            if random.random() < epsilon:
                action = random.choice(valid_cols)
            else:
                with torch.no_grad():
                    q_values = brain(state)
                    for c in range(game.cols):
                        if c not in valid_cols: q_values[c] = -9999
                    action = torch.argmax(q_values).item()

            # 2. Play Move
            reward = 0
            done = False
            
            # (Logic to determine reward/game_over remains the same...)
            if game.drop_piece(action, 1): 
                if game.check_winner(1):
                    reward = 10
                    done = True
                else:
                    p2_cols = [c for c in range(game.cols) if game.board[0][c] == 0]
                    if not p2_cols:
                        done = True # Draw
                    else:
                        p2_action = random.choice(p2_cols)
                        game.drop_piece(p2_action, 2)
                        if game.check_winner(2):
                            reward = -10
                            done = True
                        elif not any(0 in row for row in game.board):
                            done = True
            else:
                reward = -100
                done = True
            
            next_state = board_to_tensor(game.board)

            # --- NEW: SAVE TO MEMORY ---
            # We don't learn yet. We just remember.
            memory.push(state, action, reward, next_state, done)
            
            state = next_state
            
            # --- NEW: LEARN FROM MEMORY ---
            # Only learn if we have enough examples
            if len(memory) > batch_size:
                # 1. Get a random batch
                states, actions, rewards, next_states, dones = memory.sample(batch_size)
                
                # Convert lists to Tensors (The brain needs Tensors)
                # stack() creates a batch (e.g., 64 boards at once)
                t_states = torch.stack(states)
                t_next_states = torch.stack(next_states)
                t_rewards = torch.tensor(rewards, dtype=torch.float32)
                t_dones = torch.tensor(dones, dtype=torch.float32)
                t_actions = torch.tensor(actions)
                
                # 2. Calculate Targets for the WHOLE BATCH at once
                with torch.no_grad():
                    # We want the max score for each of the 64 next states
                    next_max = torch.max(brain(t_next_states), dim=1)[0]
                    # If game is done, future score is 0. (1 - t_dones) handles this.
                    targets = t_rewards + (0.9 * next_max * (1 - t_dones))
                
                # 3. Calculate Predictions
                # This gathers the Q-value for the specific action we took
                current_q = brain(t_states).gather(1, t_actions.unsqueeze(1)).squeeze(1)
                
                # 4. Update
                loss = loss_fn(current_q, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if done:
                game_over = True

        if epsilon > min_epsilon:
            epsilon *= decay
            
        if episode % 100 == 0:
            print(f"Episode {episode} | Epsilon: {epsilon:.2f}")

    print("Training Complete!")
    return brain

if __name__ == "__main__":
    trained_brain = train_dqn()
    
    # Save the brain so we can use it later
    torch.save(trained_brain.state_dict(), "connect4_brain.pth")
    print("Brain saved to 'connect4_brain.pth'")