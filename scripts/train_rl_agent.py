#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import time
import argparse
from datetime import datetime

# Import ROS messages
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped

# Import our RL modules
from f1tenth_rl.environment import F1TenthEnv
from f1tenth_rl.models.dqn import DQNAgent
from f1tenth_rl.models.ppo import PPOAgent
from f1tenth_rl.utils.rewards import calculate_reward

class TrainingNode(Node):
    def __init__(self, args):
        super().__init__('training_node')
        
        # Parse arguments
        self.algorithm = args.algorithm
        self.episodes = args.episodes
        self.save_dir = args.save_dir
        self.eval_interval = args.eval_interval
        self.load_model = args.load_model
        
        # Create save directory
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Publishers
        self.drive_pub = self.create_publisher(
            AckermannDriveStamped, '/drive', 10)
        
        # Subscribers
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10)
        
        self.odom_sub = self.create_subscription(
            Odometry, '/ego_racecar/odom', self.odom_callback, 10)
        
        # Store observations
        self.latest_scan = None
        self.latest_odom = None
        self.prev_odom = None
        
        # Environment setup
        self.env = F1TenthEnv()
        
        # Define state and action dimensions
        self.state_dim = 1080  # Laser scan dimensions
        self.action_dim = 2    # Steering angle and velocity
        
        # Initialize agent
        if self.algorithm == 'dqn':
            self.agent = DQNAgent(
                state_dim=self.state_dim,
                action_dim=15,  # Discretized action space
                hidden_dim=256,
                learning_rate=3e-4
            )
            
            # Define discrete action space
            self.actions = []
            for steering in np.linspace(-0.4, 0.4, 5):
                for velocity in [1.0, 2.0, 3.0]:
                    self.actions.append([steering, velocity])
                    
        elif self.algorithm == 'ppo':
            self.agent = PPOAgent(
                state_dim=self.state_dim,
                action_dim=self.action_dim,
                hidden_dim=256,
                lr=3e-4
            )
        else:
            self.get_logger().error(f"Unsupported algorithm: {self.algorithm}")
            return
        
        # Load model if specified
        if self.load_model:
            try:
                self.agent.load(self.load_model)
                self.get_logger().info(f"Loaded model from {self.load_model}")
            except Exception as e:
                self.get_logger().error(f"Failed to load model: {e}")
        
        # Training metrics
        self.episode_rewards = []
        self.episode_lengths = []
        self.eval_rewards = []
        
        # Training variables
        self.current_episode = 0
        self.current_step = 0
        self.episode_reward = 0.0
        self.episode_started = False
        
        self.get_logger().info(f"Training node initialized with {self.algorithm} algorithm")
        self.get_logger().info(f"Will train for {self.episodes} episodes")
        
        # Start training loop
        self.timer = self.create_timer(0.1, self.training_step)
    
    def scan_callback(self, msg):
        """Store laser scan data"""
        self.latest_scan = msg
    
    def odom_callback(self, msg):
        """Store odometry data"""
        self.prev_odom = self.latest_odom
        self.latest_odom = msg
    
    def get_state(self):
        """Process laser scan into state representation"""
        if self.latest_scan is None:
            return None
            
        ranges = np.array(self.latest_scan.ranges)
        ranges[np.isinf(ranges)] = 10.0
        normalized_ranges = ranges / 10.0
        
        return normalized_ranges
    
    def publish_drive_command(self, steering, velocity):
        """Send drive command to the car"""
        msg = AckermannDriveStamped()
        msg.drive.steering_angle = float(steering)
        msg.drive.speed = float(velocity)
        self.drive_pub.publish(msg)
    
    def training_step(self):
        """Execute one step of the training loop"""
        # Wait for simulator data
        if self.latest_scan is None or self.latest_odom is None:
            return
        
        # Start new episode if needed
        if not self.episode_started:
            self.episode_started = True
            self.episode_reward = 0.0
            self.current_step = 0
            self.get_logger().info(f"Starting episode {self.current_episode + 1}")
            
            # Reset simulator by publishing zero speed
            self.publish_drive_command(0.0, 0.0)
            time.sleep(1.0)  # Let simulator reset
            
            return
        
        # Get current state
        current_state = self.get_state()
        if current_state is None:
            return
            
        # Select and execute action
        if self.algorithm == 'dqn':
            # Epsilon-greedy policy
            if np.random.random() < self.agent.epsilon:
                action_idx = np.random.randint(0, len(self.actions))
            else:
                state_tensor = torch.FloatTensor(current_state).unsqueeze(0)
                action_idx = self.agent.select_action(state_tensor)
                
            steering, velocity = self.actions[action_idx]
            self.publish_drive_command(steering, velocity)
            
        elif self.algorithm == 'ppo':
            # Sample from policy
            state_tensor = torch.FloatTensor(current_state).unsqueeze(0)
            action, log_prob, value = self.agent.select_action(state_tensor)
            
            steering, velocity = action
            self.publish_drive_command(steering, velocity)
        
        # Wait for next observation
        if self.prev_odom is None:
            return
            
        # Calculate reward and check for done condition
        reward, done = calculate_reward(
            self.latest_scan,
            self.latest_odom,
            self.prev_odom
        )
        
        # Update step counts
        self.current_step += 1
        self.episode_reward += reward
        
        # Store transition
        next_state = self.get_state()
        
        if self.algorithm == 'dqn':
            self.agent.store_transition(current_state, action_idx, reward, next_state, done)
            
            # Update model
            if len(self.agent.replay_buffer) > self.agent.batch_size:
                loss = self.agent.train()
                
        elif self.algorithm == 'ppo':
            self.agent.store_transition(current_state, action, log_prob, reward, value, done)
            
            # Update PPO after collecting enough experience
            if self.current_step % 200 == 0:
                next_value = 0.0
                if not done:
                    _, _, next_value = self.agent.select_action(
                        torch.FloatTensor(next_state).unsqueeze(0)
                    )
                loss = self.agent.train(next_value)
        
        # Episode end handling
        if done or self.current_step >= 1000:
            self.episode_rewards.append(self.episode_reward)
            self.episode_lengths.append(self.current_step)
            
            # Log episode results
            self.get_logger().info(
                f"Episode {self.current_episode + 1} finished: "
                f"Reward={self.episode_reward:.2f}, "
                f"Steps={self.current_step}, "
                f"Epsilon={getattr(self.agent, 'epsilon', 'N/A')}"
            )
            
            # Save model periodically
            if (self.current_episode + 1) % 10 == 0:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = os.path.join(
                    self.save_dir,
                    f"{self.algorithm}_episode_{self.current_episode + 1}_{timestamp}.pt"
                )
                self.agent.save(save_path)
                self.get_logger().info(f"Saved model to {save_path}")
                
                # Plot training progress
                self.plot_progress()
            
            # Evaluation
            if (self.current_episode + 1) % self.eval_interval == 0:
                self.evaluate()
            
            # Reset for next episode
            self.current_episode += 1
            self.episode_started = False
            
            # Check if training is complete
            if self.current_episode >= self.episodes:
                self.get_logger().info("Training complete!")
                
                # Save final model
                final_path = os.path.join(
                    self.save_dir,
                    f"{self.algorithm}_final.pt"
                )
                self.agent.save(final_path)
                self.get_logger().info(f"Saved final model to {final_path}")
                
                # Plot final progress
                self.plot_progress()
                
                # Stop training loop
                self.timer.cancel()
    
    def evaluate(self, num_episodes=3):
        """Evaluate the current agent"""
        self.get_logger().info("Starting evaluation...")
        
        eval_rewards = []
        
        for i in range(num_episodes):
            # Reset environment
            self.publish_drive_command(0.0, 0.0)
            time.sleep(1.0)
            
            # Run one episode with deterministic policy
            ep_reward = 0.0
            ep_steps = 0
            done = False
            
            while not done and ep_steps < 1000:
                # Get state
                state = self.get_state()
                if state is None:
                    continue
                
                # Select action deterministically
                if self.algorithm == 'dqn':
                    state_tensor = torch.FloatTensor(state).unsqueeze(0)
                    action_idx = self.agent.select_action(state_tensor)
                    steering, velocity = self.actions[action_idx]
                    
                elif self.algorithm == 'ppo':
                    state_tensor = torch.FloatTensor(state).unsqueeze(0)
                    action, _, _ = self.agent.select_action(state_tensor, deterministic=True)
                    steering, velocity = action
                
                # Execute action
                self.publish_drive_command(steering, velocity)
                
                # Wait for next observation
                time.sleep(0.1)
                
                # Calculate reward
                if self.prev_odom is not None:
                    reward, done = calculate_reward(
                        self.latest_scan,
                        self.latest_odom,
                        self.prev_odom
                    )
                    ep_reward += reward
                
                ep_steps += 1
            
            self.get_logger().info(f"Eval episode {i+1}: Reward={ep_reward:.2f}, Steps={ep_steps}")
            eval_rewards.append(ep_reward)
        
        # Store evaluation results
        avg_reward = sum(eval_rewards) / len(eval_rewards)
        self.eval_rewards.append(avg_reward)
        self.get_logger().info(f"Evaluation complete. Average reward: {avg_reward:.2f}")
    
    def plot_progress(self):
        """Plot and save training progress"""
        # Create plot directory
        plot_dir = os.path.join(self.save_dir, 'plots')
        os.makedirs(plot_dir, exist_ok=True)
        
        # Plot episode rewards
        plt.figure(figsize=(10, 6))
        plt.plot(self.episode_rewards)
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.title(f'{self.algorithm} Training Progress')
        plt.savefig(os.path.join(plot_dir, 'episode_rewards.png'))
        plt.close()
        
        # Plot moving average
        if len(self.episode_rewards) > 10:
            moving_avg = np.convolve(self.episode_rewards, np.ones(10)/10, mode='valid')
            plt.figure(figsize=(10, 6))
            plt.plot(moving_avg)
            plt.xlabel('Episode')
            plt.ylabel('Average Reward (10 episodes)')
            plt.title(f'{self.algorithm} Moving Average Reward')
            plt.savefig(os.path.join(plot_dir, 'moving_average.png'))
            plt.close()
        
        # Plot evaluation rewards
        if self.eval_rewards:
            plt.figure(figsize=(10, 6))
            plt.plot(np.arange(self.eval_interval, 
                              len(self.eval_rewards) * self.eval_interval + 1, 
                              self.eval_interval), 
                    self.eval_rewards)
            plt.xlabel('Episode')
            plt.ylabel('Evaluation Reward')
            plt.title(f'{self.algorithm} Evaluation Performance')
            plt.savefig(os.path.join(plot_dir, 'eval_rewards.png'))
            plt.close()
        
        # Save data
        np.savetxt(os.path.join(plot_dir, 'episode_rewards.txt'), self.episode_rewards)
        np.savetxt(os.path.join(plot_dir, 'episode_lengths.txt'), self.episode_lengths)
        if self.eval_rewards:
            np.savetxt(os.path.join(plot_dir, 'eval_rewards.txt'), self.eval_rewards)


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train RL agent for F1TENTH')
    parser.add_argument('--algorithm', type=str, default='dqn', choices=['dqn', 'ppo'],
                        help='RL algorithm to use (dqn or ppo)')
    parser.add_argument('--episodes', type=int, default=200,
                        help='Number of episodes to train')
    parser.add_argument('--save-dir', type=str, default='models',
                        help='Directory to save models and plots')
    parser.add_argument('--eval-interval', type=int, default=20,
                        help='Evaluate every N episodes')
    parser.add_argument('--load-model', type=str, default='',
                        help='Path to model to load (empty for training from scratch)')
    
    args = parser.parse_args()
    
    # Initialize ROS
    rclpy.init()
    
    # Create and run training node
    node = TrainingNode(args)
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Make sure to clean up
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
