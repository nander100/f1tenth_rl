#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np

class ActorCritic(nn.Module):
    """
    Actor-Critic Network for PPO
    """
    def __init__(self, state_dim, action_dim, hidden_dim=256, use_farthest_point=True):
        super(ActorCritic, self).__init__()
        
        # Add input dimensions for farthest point features if enabled
        self.use_farthest_point = use_farthest_point
        input_dim = state_dim
        if use_farthest_point:
            # Add 2 more dimensions for steering suggestion and distance
            input_dim += 2
        
        # Shared feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Actor (policy) network
        self.actor_mean = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        # Actor log std (learnable)
        self.actor_log_std = nn.Parameter(torch.zeros(1, action_dim))
        
        # Critic (value) network
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state):
        """Forward pass through the network"""
        features = self.feature_extractor(state)
        
        # Actor: action mean and standard deviation
        action_mean = self.actor_mean(features)
        action_std = torch.exp(self.actor_log_std)
        
        # Critic: state value
        value = self.critic(features)
        
        return action_mean, action_std, value
    
    def get_action(self, state, deterministic=False):
        """Sample action from the policy distribution"""
        action_mean, action_std, _ = self.forward(state)
        
        # Create normal distribution
        dist = Normal(action_mean, action_std)
        
        # Sample from distribution or take mean
        if deterministic:
            action = action_mean
        else:
            action = dist.sample()
        
        # Calculate log probability of the action
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        
        # Squash action to appropriate range (for F1TENTH: steering and velocity)
        # For steering: [-0.4, 0.4] radians
        # For velocity: [0, 3.0] m/s
        action_squashed = torch.zeros_like(action)
        action_squashed[:, 0] = torch.tanh(action[:, 0]) * 0.4  # Steering
        action_squashed[:, 1] = torch.sigmoid(action[:, 1]) * 3.0  # Velocity
        
        return action_squashed, log_prob

class PPOAgent:
    """
    Proximal Policy Optimization agent for F1TENTH
    """
    def __init__(self, state_dim, action_dim, hidden_dim=256, lr=3e-4, use_farthest_point=True):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.use_farthest_point = use_farthest_point
        
        # Actor-Critic network
        self.actor_critic = ActorCritic(state_dim, action_dim, hidden_dim, use_farthest_point)
        
        # Optimizer
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=lr)
        
        # Hyperparameters
        self.gamma = 0.99  # Discount factor
        self.gae_lambda = 0.95  # GAE parameter
        self.clip_ratio = 0.2  # PPO clip ratio
        self.max_grad_norm = 0.5  # Gradient clipping
        self.value_coef = 0.5  # Value loss coefficient
        self.entropy_coef = 0.01  # Entropy coefficient
        
        # Training data
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
    
    def select_action(self, state, deterministic=False):
        """Select action from the policy"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action, log_prob = self.actor_critic.get_action(state_tensor, deterministic)
            _, _, value = self.actor_critic(state_tensor)
        
        # Convert to numpy
        action_np = action.squeeze(0).numpy()
        log_prob_np = log_prob.squeeze(0).numpy()
        value_np = value.squeeze(0).numpy()
        
        return action_np, log_prob_np, value_np
    
    def store_transition(self, state, action, log_prob, reward, value, done):
        """Store transition in memory"""
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)
    
    def train(self, next_value=0):
        """Train PPO agent using collected trajectories"""
        # Convert lists to tensors
        states = torch.FloatTensor(np.array(self.states))
        actions = torch.FloatTensor(np.array(self.actions))
        old_log_probs = torch.FloatTensor(np.array(self.log_probs))
        rewards = torch.FloatTensor(np.array(self.rewards))
        values = torch.FloatTensor(np.array(self.values))
        dones = torch.FloatTensor(np.array(self.dones))
        
        # Compute returns and advantages (Generalized Advantage Estimation)
        returns, advantages = self._compute_gae(rewards, values, dones, next_value)
        
        # PPO update (multiple epochs)
        for _ in range(10):  # Number of epochs
            # Forward pass through the network
            action_means, action_stds, values_pred = self.actor_critic(states)
            
            # Calculate action log probabilities
            dist = Normal(action_means, action_stds)
            new_log_probs = dist.log_prob(actions).sum(dim=-1, keepdim=True)
            
            # Calculate entropy
            entropy = dist.entropy().mean()
            
            # Calculate ratio (pi_theta / pi_theta_old)
            ratio = torch.exp(new_log_probs - old_log_probs)
            
            # Calculate surrogate losses
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * advantages
            
            # Calculate actor loss (negative for gradient ascent)
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # Calculate critic loss
            value_loss = F.mse_loss(values_pred, returns)
            
            # Total loss
            loss = actor_loss + self.value_coef * value_loss - self.entropy_coef * entropy
            
            # Update network parameters
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.optimizer.step()
        
        # Clear memory after update
        self._clear_memory()
        
        return loss.item()
    
    def _compute_gae(self, rewards, values, dones, next_value=0):
        """Compute returns and advantages using Generalized Advantage Estimation"""
        returns = torch.zeros_like(rewards)
        advantages = torch.zeros_like(rewards)
        
        # Last value if episode didn't end
        last_gae_lam = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_val = next_value
            else:
                next_val = values[t + 1]
            
            # Calculate TD error and GAE
            delta = rewards[t] + self.gamma * next_val * (1 - dones[t]) - values[t]
            last_gae_lam = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_gae_lam
            
            advantages[t] = last_gae_lam
            returns[t] = advantages[t] + values[t]
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return returns, advantages
    
    def _clear_memory(self):
        """Clear trajectory memory"""
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
    
    def save(self, path):
        """Save model to file"""
        torch.save({
            'actor_critic_state_dict': self.actor_critic.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'use_farthest_point': self.use_farthest_point
        }, path)
    
    def load(self, path):
        """Load model from file"""
        checkpoint = torch.load(path)
        
        # Check if the saved model uses farthest point
        saved_use_farthest_point = checkpoint.get('use_farthest_point', False)
        
        # Only load if configurations match or handle the mismatch
        if saved_use_farthest_point == self.use_farthest_point:
            self.actor_critic.load_state_dict(checkpoint['actor_critic_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        else:
            raise ValueError(
                f"Model mismatch: Saved model use_farthest_point={saved_use_farthest_point}, "
                f"but current setting is {self.use_farthest_point}"
            )
