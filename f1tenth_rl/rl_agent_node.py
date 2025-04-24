#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
import numpy as np
import torch
import os
import yaml
from datetime import datetime
from f1tenth_rl.environment import F1TenthEnv
from f1tenth_rl.models.dqn import DQNAgent
from f1tenth_rl.utils.rewards import calculate_reward

class RLAgentNode(Node):
    def __init__(self):
        super().__init__('rl_agent_node')
        
        # Declare parameters
        self.declare_parameter('training_mode', True)
        self.declare_parameter('model_type', 'dqn')
        self.declare_parameter('model_path', '')
        self.declare_parameter('save_path', 'models/')
        self.declare_parameter('start_x', 0.0)
        self.declare_parameter('start_y', 0.0)
        self.declare_parameter('start_yaw', 0.0)
        
        # Get parameters
        self.training_mode = self.get_parameter('training_mode').value
        self.model_type = self.get_parameter('model_type').value
        self.model_path = self.get_parameter('model_path').value
        self.save_path = self.get_parameter('save_path').value
        
        # Create directories if they don't exist
        os.makedirs(self.save_path, exist_ok=True)
        
        # Publishers and subscribers
        self.drive_pub = self.create_publisher(
            AckermannDriveStamped, '/drive', 10)
        
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10)
        
        self.odom_sub = self.create_subscription(
            Odometry, '/ego_racecar/odom', self.odom_callback, 10)
        
        # Store the latest observations
        self.latest_scan = None
        self.latest_odom = None
        self.prev_odom = None
        
        # Initialize environment with reference to this node
        self.env = F1TenthEnv(node=self)
        
        # Set starting position from parameters
        self.env.start_position = [
            self.get_parameter('start_x').value,
            self.get_parameter('start_y').value,
            self.get_parameter('start_yaw').value
        ]
        
        # Initialize RL agent
        self.initialize_agent()
        
        # Training loop timer (runs at 10Hz)
        self.timer = self.create_timer(0.1, self.training_loop)
        
        # Episodic data
        self.episode_reward = 0.0
        self.episode_steps = 0
        self.episodes_completed = 0
        self.max_steps_per_episode = 1000
        
        self.get_logger().info('RL Agent Node initialized')
    
    def initialize_agent(self):
        """Initialize the RL agent based on specified model type"""
        state_dim = 1080  # Laser scan dimensions
        action_dim = 2    # Steering angle and velocity
        
        if self.model_type == 'dqn':
            # Discretized actions for DQN
            # We'll use 5 steering angles and 3 velocities
            self.agent = DQNAgent(
                state_dim=state_dim,
                action_dim=15,  # 5x3 possible actions
                hidden_dim=256,
                learning_rate=3e-4
            )
            
            # Define our discrete action space
            self.actions = []
            for steering in np.linspace(-0.4, 0.4, 5):  # -0.4 to 0.4 rad
                for velocity in [1.0, 2.0, 3.0]:  # m/s
                    self.actions.append([steering, velocity])
            
            self.get_logger().info(f'Initialized DQN agent with {len(self.actions)} discrete actions')
        
        # Load model if provided
        if self.model_path:
            try:
                self.agent.load(self.model_path)
                self.get_logger().info(f'Loaded model from {self.model_path}')
            except Exception as e:
                self.get_logger().error(f'Failed to load model: {e}')
    
    def scan_callback(self, msg):
        """Store the latest laser scan data"""
        self.latest_scan = msg
    
    def odom_callback(self, msg):
        """Store the latest odometry data"""
        self.prev_odom = self.latest_odom
        self.latest_odom = msg
    
    def get_state(self):
        """Convert laser scan to state vector for RL agent"""
        if self.latest_scan is None:
            return None
            
        # Extract ranges from laser scan
        ranges = np.array(self.latest_scan.ranges)
        
        # Replace inf values with a large number
        ranges[np.isinf(ranges)] = 10.0
        
        # Normalize ranges to [0, 1]
        normalized_ranges = ranges / 10.0
        
        return normalized_ranges
    
    def publish_drive_command(self, steering, velocity):
        """Publish drive command to the car"""
        msg = AckermannDriveStamped()
        msg.drive.steering_angle = float(steering)
        msg.drive.speed = float(velocity)
        self.drive_pub.publish(msg)
    
    def training_loop(self):
        """Main RL training/inference loop"""
        if self.latest_scan is None or self.latest_odom is None:
            return
        
        # Get current state
        current_state = self.get_state()
        
        if current_state is None:
            return
            
        # In training mode, use epsilon-greedy policy
        if self.training_mode:
            # Select action (epsilon-greedy)
            if np.random.random() < self.agent.epsilon:
                # Random action
                action_idx = np.random.randint(0, len(self.actions))
            else:
                # Model prediction
                state_tensor = torch.FloatTensor(current_state).unsqueeze(0)
                action_idx = self.agent.select_action(state_tensor)
                
            # Get the actual steering and velocity values
            steering, velocity = self.actions[action_idx]
            
        # In testing mode, use greedy policy
        else:
            state_tensor = torch.FloatTensor(current_state).unsqueeze(0)
            action_idx = self.agent.select_action(state_tensor)
            steering, velocity = self.actions[action_idx]
        
        # Execute action
        self.publish_drive_command(steering, velocity)
        
        # Wait for next observation (handled by callbacks)
        
        # If we have a previous state, calculate reward and store transition
        if self.prev_odom is not None and self.training_mode:
            # Calculate reward
            reward, done = calculate_reward(
                self.latest_scan,
                self.latest_odom,
                self.prev_odom
            )
            
            # Accumulate episode reward
            self.episode_reward += reward
            self.episode_steps += 1
            
            # Store transition in replay buffer
            next_state = self.get_state()
            self.agent.store_transition(current_state, action_idx, reward, next_state, done)
            
            # Train the agent
            if len(self.agent.replay_buffer) > self.agent.batch_size:
                self.agent.train()
            
            # Check if episode is done
            if done or self.episode_steps >= self.max_steps_per_episode:
                self.episodes_completed += 1
                self.env.reset_car_position()
                self.get_logger().info("Episode ended! Resetting car position.")
                
                # Log episode stats
                self.get_logger().info(
                    f'Episode {self.episodes_completed}: '
                    f'Reward={self.episode_reward:.2f}, '
                    f'Steps={self.episode_steps}'
                )
                
                # Save model periodically
                if self.episodes_completed % 10 == 0:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    save_path = os.path.join(
                        self.save_path, 
                        f'{self.model_type}_episode_{self.episodes_completed}_{timestamp}.pt'
                    )
                    self.agent.save(save_path)
                    self.get_logger().info(f'Saved model to {save_path}')
                
                # Reset episode stats
                self.episode_reward = 0.0
                self.episode_steps = 0
                
                # Reduce exploration over time
                self.agent.epsilon = max(
                    self.agent.epsilon * 0.99,  # Decay rate
                    0.05  # Minimum exploration
                )

def main(args=None):
    rclpy.init(args=args)
    node = RLAgentNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
