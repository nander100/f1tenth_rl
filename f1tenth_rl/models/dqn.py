#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

class DQNNetwork(nn.Module):
    """
    Deep Q-Network for F1TENTH reinforcement learning
    """
    def __init__(self, state_dim, action_dim, hidden_dim=256, use_farthest_point=True):
        super(DQNNetwork, self).__init__()
        
        # Add input dimensions for farthest point features if enabled
        self.use_farthest_point = use_farthest_point
        input_dim = state_dim
        if use_farthest_point:
            # Add 2 more dimensions for steering suggestion and distance
            input_dim += 2
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, x):
        return self.network(x)

class DQNAgent:
    """
    DQN Agent implementation for the F1TENTH simulator
    """
    def __init__(self, state_dim, action_dim, hidden_dim=256, learning_rate=3e-4, use_farthest_point=True):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.use_farthest_point = use_farthest_point
        
        # Create Q-networks (online and target)
        self.q_network = DQNNetwork(state_dim, action_dim, hidden_dim, use_farthest_point)
        self.target_network = DQNNetwork(state_dim, action_dim, hidden_dim, use_farthest_point)
        
        # Copy parameters from online to target network
        self.update_target_network()
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Loss function
        self.loss_fn = nn.MSELoss()
        
        # Replay buffer
        self.replay_buffer = deque(maxlen=10000)
        
        # Hyperparameters
        self.batch_size = 64
        self.gamma = 0.99  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        self.target_update_freq = 10  # Update target network every N steps
        self.update_counter = 0
    
    def update_target_network(self):
        """Copy parameters from online to target network"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def select_action(self, state):
        """Select action using epsilon-greedy policy"""
        # Pass state through Q-network to get Q-values for all actions
        with torch.no_grad():
            q_values = self.q_network(state)
        
        # Select action with highest Q-value
        return torch.argmax(q_values).item()
    
    def store_transition(self, state, action, reward, next_state, done):
        """Store transition in replay buffer"""
        self.replay_buffer.append((state, action, reward, next_state, done))
    
    def train(self):
        """Train the DQN agent using experience replay"""
        # Sample mini-batch from replay buffer
        batch = random.sample(self.replay_buffer, self.batch_size)
        
        # Unpack batch
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(np.array(actions))
        rewards = torch.FloatTensor(np.array(rewards))
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(np.array(dones))
        
        # Get current Q-values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Compute target Q-values
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Compute loss
        loss = self.loss_fn(current_q_values, target_q_values)
        
        # Update online network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network periodically
        self.update_counter += 1
        if self.update_counter % self.target_update_freq == 0:
            self.update_target_network()
            self.update_counter = 0
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        return loss.item()
    
    def save(self, path):
        """Save model to file"""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'use_farthest_point': self.use_farthest_point
        }, path)
    
    def load(self, path):
        """Load model from file"""
        checkpoint = torch.load(path)
        
        # Check if the saved model uses farthest point
        saved_use_farthest_point = checkpoint.get('use_farthest_point', False)
        
        # Only load if configurations match or handle the mismatch
        if saved_use_farthest_point == self.use_farthest_point:
            self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
            self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint['epsilon']
        else:
            raise ValueError(
                f"Model mismatch: Saved model use_farthest_point={saved_use_farthest_point}, "
                f"but current setting is {self.use_farthest_point}"
            )
