import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from replay_buffer import ReplayMemory, Transition
import numpy as np
from time import sleep

# Neural Network class
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x


class DQNAgent:
    def __init__(self, state_size, action_size, device):
        self.state_size = state_size
        self.action_size = action_size
        self.device = device
        
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.995
        self.learning_rate = 1e-3
        self.batch_size = 64
        self.target_update_freq = 1000
        
        # Initialize policy network and target network
        self.policy_net = DQN(input_dim=state_size, output_dim=action_size).to(device)
        self.target_net = DQN(input_dim=state_size, output_dim=action_size).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        
        # Replay memory
        self.memory = ReplayMemory(100000)
        
        # Steps counter
        self.steps_done = 0
    
    def save_model(self, path):
        torch.save(self.policy_net.state_dict(), path)

    def load_model(self, path):
        self.policy_net.load_state_dict(torch.load(path))

    def select_action(self, state, legal_actions):
        """
        Select an action using epsilon-greedy policy.

        Args:
            state (np.array): The current state.
            legal_actions (np.array): Binary array indicating which actions are legal.

        Returns:
            int: The selected action.
        """
        if np.random.rand() <= self.epsilon:
            # Exploration: Sample a random legal action
            legal_actions = np.where(legal_actions == 1)[0]
            action = np.random.choice(legal_actions)
        else:
            state = torch.FloatTensor(state).to(self.device).unsqueeze(0)  # Add batch dimension
            with torch.no_grad():
                q_values = self.policy_net(state)

            # Mask illegal actions by setting their Q-values to a very low number
            q_values[0][~torch.tensor(legal_actions, dtype=torch.bool)] = float('-inf')

            # Choose the action with the highest Q-value among legal actions
            action = q_values.argmax().item()

        return action
    
    # Save a transition
    def remember(self, state, action, next_state, reward, done):
        self.memory.push(state, action, next_state, reward, done)
    
    # Fit the policy network to the target network
    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return

        # Sample a batch of transitions
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        # Convert batch data to tensors
        state_batch = torch.FloatTensor(np.array(batch.state)).to(self.device)
        action_batch = torch.LongTensor(batch.action).to(self.device).unsqueeze(1)
        reward_batch = torch.FloatTensor(batch.reward).to(self.device).unsqueeze(1)
        next_state_batch = torch.FloatTensor(np.array(batch.next_state)).to(self.device)

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        next_q_values = self.target_net(next_state_batch).max(1)[0].unsqueeze(1)

        # Compute the expected Q values
        expected_state_action_values = (next_q_values * self.gamma) + reward_batch

        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)
    
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def act_deterministic(self, state, legal_actions):
        # Convert state to a PyTorch tensor
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)  # Add batch dimension
        with torch.no_grad():
            q_values = self.policy_net(state)  # Forward pass
        q_values[0][~torch.tensor(legal_actions, dtype=torch.bool)] = float('-inf')
        return q_values.argmax().item()  # Select the action with the highest Q-value