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
from f1tenth_rl.utils.farthest_point import calculate_farthest_point_steering

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
        self.declare_parameter('use_farthest_point', True)
        self.declare_parameter('farthest_point_noise', 0.1)
        self.declare_parameter('farthest_point_weight', 1.0)
        
        # Get parameters
        self.training_mode = self.get_parameter('training_mode').value
        self.model_type = self.get_parameter('model_type').value
        self.model_path = self.get_parameter('model_path').value
        self.save_path = self.get_parameter('save_path').value
        self.use_farthest_point = self.get_parameter('use_farthest_point').value
        self.farthest_point_noise = self.get_parameter('farthest_point_noise').value
        self.farthest_point_weight = self.get_parameter('farthest_point_weight').value
        
        self.get_logger().info(f"Using farthest point feature: {self.use_farthest_point}")
        if self.use_farthest_point:
            self.get_logger().info(f"Farthest point noise level: {self.farthest_point_noise}")
            self.get_logger().info(f"Farthest point weight: {self.farthest_point_weight}")
        
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
        self.env = F1TenthEnv(
            node=self,
            use_farthest_point=self.use_farthest_point,
            farthest_point_noise=self.farthest_point_noise
        )
        
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
        
        # Farthest point data
        self.farthest_steering = 0.0
        self.farthest_distance = 0.0
        
        self.get_logger().info('RL Agent Node initialized')
    
    def initialize_agent(self):
        """Initialize the RL agent based on specified model type"""
        # Base state dimension (laser scan)
        state_dim = 1080  # Laser scan dimensions
        
        # Adjust state dimension if using farthest point
        # Add 2 dimensions for steering suggestion and distance
        if self.use_farthest_point:
            state_dim_with_fp = state_dim + 2
        else:
            state_dim_with_fp = state_dim
            
        action_dim = 2    # Steering angle and velocity
        
        if self.model_type == 'dqn':
            # Discretized actions for DQN
            # We'll use 5 steering angles and 3 velocities
            self.agent = DQNAgent(
                state_dim=state_dim,
                action_dim=15,  # 5x3 possible actions
                hidden_dim=256,
                learning_rate=3e-4,
                use_farthest_point=self.use_farthest_point
            )
            
            # Define our discrete action space
            self.actions = []
            for steering in np.linspace(-0.4, 0.4, 5):  # -0.4 to 0.4 rad
                for velocity in [1.0, 2.0, 3.0]:  # m/s
                    self.actions.append([steering, velocity])
            
            self.get_logger().info(f'Initialized DQN agent with {len(self.actions)} discrete actions')
        elif self.model_type == 'ppo':
            # Import PPO if it's selected
            from f1tenth_rl.models.ppo import PPOAgent
            
            # Initialize PPO agent
            self.agent = PPOAgent(
                state_dim=state_dim,
                action_dim=action_dim,
                hidden_dim=256,
                lr=3e-4,
                use_farthest_point=self.use_farthest_point
            )
            
            self.get_logger().info('Initialized PPO agent')
        
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
        
        # Calculate farthest point steering if enabled
        if self.use_farthest_point and self.latest_scan is not None:
            self.farthest_steering, self.farthest_distance = calculate_farthest_point_steering(
                self.latest_scan,
                noise_level=0.0  # Only add noise during training
            )
    
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
        
        # Add farthest point information if enabled
        if self.use_farthest_point:
            # Normalize steering to [-1, 1] and distance to [0, 1]
            normalized_steering = self.farthest_steering / 0.4
            normalized_distance = self.farthest_distance / 10.0
            
            # Concatenate with scan data
            return np.append(normalized_ranges, [normalized_steering, normalized_distance])
        else:
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
            if self.model_type == 'dqn':
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
            elif self.model_type == 'ppo':
                # Sample from policy
                state_tensor = torch.FloatTensor(current_state).unsqueeze(0)
                action, log_prob, value = self.agent.select_action(state_tensor)
                
                steering, velocity = action
                
        # In testing mode, use greedy policy
        else:
            if self.model_type == 'dqn':
                state_tensor = torch.FloatTensor(current_state).unsqueeze(0)
                action_idx = self.agent.select_action(state_tensor)
                steering, velocity = self.actions[action_idx]
            elif self.model_type == 'ppo':
                state_tensor = torch.FloatTensor(current_state).unsqueeze(0)
                action, _, _ = self.agent.select_action(state_tensor, deterministic=True)
                steering, velocity = action
        
        # Blend with farthest point steering if desired
        if not self.training_mode and self.use_farthest_point and self.farthest_point_weight > 0:
            # Blend RL steering with farthest point steering
            blended_steering = (1 - self.farthest_point_weight) * steering + self.farthest_point_weight * self.farthest_steering
            # Clip to valid range
            blended_steering = np.clip(blended_steering, -0.4, 0.4)
            steering = blended_steering
        
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
            
            # Add farthest point following reward if enabled
            if self.use_farthest_point and self.farthest_point_weight > 0:
                # Reward for following the farthest point suggestion
                # Lower penalty for steering close to the suggested direction
                farthest_point_reward = -self.farthest_point_weight * abs(steering - self.farthest_steering)
                reward += farthest_point_reward
            
            # Accumulate episode reward
            self.episode_reward += reward
            self.episode_steps += 1
            
            # Store transition based on agent type
            if self.model_type == 'dqn':
                next_state = self.get_state()
                self.agent.store_transition(current_state, action_idx, reward, next_state, done)
                
                # Train the agent
                if len(self.agent.replay_buffer) > self.agent.batch_size:
                    self.agent.train()
            elif self.model_type == 'ppo':
                self.agent.store_transition(current_state, action, log_prob, reward, value, done)
                
                # Update PPO after collecting enough experience
                if self.episode_steps % 200 == 0:
                    next_state = self.get_state()
                    next_value = 0.0
                    if not done and next_state is not None:
                        _, _, next_value = self.agent.select_action(
                            torch.FloatTensor(next_state).unsqueeze(0)
                        )
                    self.agent.train(next_value)
            
            # Check if episode is done
            if done or self.episode_steps >= self.max_steps_per_episode:
                self.episodes_completed += 1
                self.env.reset_car_position()
                self.get_logger().info("Episode ended! Resetting car position.")
                
                # Log episode stats
                self.get_logger().info(
                    f'Episode {self.episodes_completed}: '
                    f'Reward={self.episode_reward:.2f}, '
                    f'Steps={self.episode_steps}, '
                    f'Epsilon={getattr(self.agent, "epsilon", "N/A")}'
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
                
                # Reduce exploration over time for DQN
                if self.model_type == 'dqn':
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
