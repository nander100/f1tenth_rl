#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
import torch
import argparse
import time
import os

# Import ROS messages
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped

# Import our RL modules
from f1tenth_rl.environment import F1TenthEnv
from f1tenth_rl.models.dqn import DQNAgent
from f1tenth_rl.models.ppo import PPOAgent

class EvaluationNode(Node):
    def __init__(self, args):
        super().__init__('evaluation_node')
        
        # Parse arguments
        self.model_path = args.model_path
        self.algorithm = args.algorithm
        self.num_episodes = args.num_episodes
        self.record_data = args.record_data
        self.visualization = args.visualization
        
        if self.record_data:
            self.data_dir = os.path.join('eval_data', time.strftime("%Y%m%d-%H%M%S"))
            os.makedirs(self.data_dir, exist_ok=True)
        
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
        
        # Load model
        try:
            self.agent.load(self.model_path)
            self.get_logger().info(f"Loaded model from {self.model_path}")
        except Exception as e:
            self.get_logger().error(f"Failed to load model: {e}")
            return
        
        # Recording data
        if self.record_data:
            self.trajectory_data = []
        
        # Evaluation variables
        self.current_episode = 0
        self.episode_steps = 0
        self.episode_reward = 0.0
        self.episode_started = False
        self.all_rewards = []
        
        self.get_logger().info(f"Evaluation node initialized with {self.algorithm} algorithm")
        self.get_logger().info(f"Will evaluate for {self.num_episodes} episodes")
        
        # Start evaluation loop
        self.timer = self.create_timer(0.1, self.evaluation_step)
    
    def scan_callback(self, msg):
        """Store laser scan data"""
        self.latest_scan = msg
    
    def odom_callback(self, msg):
        """Store odometry data"""
        self.prev_odom = self.latest_odom
        self.latest_odom = msg
        
        # Record trajectory data
        if self.record_data and self.episode_started:
            position = msg.pose.pose.position
            orientation = msg.pose.pose.orientation
            linear_vel = msg.twist.twist.linear
            angular_vel = msg.twist.twist.angular
            
            self.trajectory_data.append({
                'episode': self.current_episode,
                'step': self.episode_steps,
                'x': position.x,
                'y': position.y,
                'z': position.z,
                'qx': orientation.x,
                'qy': orientation.y,
                'qz': orientation.z,
                'qw': orientation.w,
                'vx': linear_vel.x,
                'vy': linear_vel.y,
                'vz': linear_vel.z,
                'wx': angular_vel.x,
                'wy': angular_vel.y,
                'wz': angular_vel.z
            })
    
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
    
    def evaluation_step(self):
        """Execute one step of the evaluation loop"""
        # Wait for simulator data
        if self.latest_scan is None or self.latest_odom is None:
            return
        
        # Start new episode if needed
        if not self.episode_started:
            if self.current_episode >= self.num_episodes:
                # All episodes complete
                self.finalize_evaluation()
                return
                
            self.episode_started = True
            self.episode_reward = 0.0
            self.episode_steps = 0
            
            if self.record_data:
                self.trajectory_data = []
                
            self.get_logger().info(f"Starting evaluation episode {self.current_episode + 1}")
            
            # Reset simulator by publishing zero speed
            self.publish_drive_command(0.0, 0.0)
            time.sleep(1.0)  # Let simulator reset
            
            return
        
        # Get current state
        current_state = self.get_state()
        if current_state is None:
            return
            
        # Select action deterministically
        if self.algorithm == 'dqn':
            state_tensor = torch.FloatTensor(current_state).unsqueeze(0)
            action_idx = self.agent.select_action(state_tensor)
            steering, velocity = self.actions[action_idx]
            
        elif self.algorithm == 'ppo':
            state_tensor = torch.FloatTensor(current_state).unsqueeze(0)
            action, _, _ = self.agent.select_action(state_tensor, deterministic=True)
            steering, velocity = action
        
        # Execute action
        self.publish_drive_command(steering, velocity)
        
        # Update step count
        self.episode_steps += 1
        
        # Calculate reward
        if self.prev_odom is not None:
            from f1tenth_rl.utils.rewards import calculate_reward
            reward, done = calculate_reward(
                self.latest_scan,
                self.latest_odom,
                self.prev_odom
            )
            self.episode_reward += reward
            
            # Check for episode end
            if done or self.episode_steps >= 1000:
                self.get_logger().info(
                    f"Episode {self.current_episode + 1} finished: "
                    f"Reward={self.episode_reward:.2f}, "
                    f"Steps={self.episode_steps}"
                )
                
                self.all_rewards.append(self.episode_reward)
                
                # Save trajectory data
                if self.record_data:
                    self.save_trajectory_data()
                
                # Reset for next episode
                self.current_episode += 1
                self.episode_started = False
    
    def save_trajectory_data(self):
        """Save trajectory data to file"""
        if not self.trajectory_data:
            return
            
        import pandas as pd
        
        # Convert to DataFrame
        df = pd.DataFrame(self.trajectory_data)
        
        # Save to CSV
        filename = os.path.join(self.data_dir, f"episode_{self.current_episode}.csv")
        df.to_csv(filename, index=False)
        self.get_logger().info(f"Saved trajectory data to {filename}")
    
    def finalize_evaluation(self):
        """Finalize evaluation and print results"""
        self.get_logger().info("Evaluation complete!")
        
        # Calculate statistics
        avg_reward = sum(self.all_rewards) / len(self.all_rewards)
        std_reward = np.std(self.all_rewards)
        min_reward = min(self.all_rewards)
        max_reward = max(self.all_rewards)
        
        self.get_logger().info(f"Results over {self.num_episodes} episodes:")
        self.get_logger().info(f"Average reward: {avg_reward:.2f} ± {std_reward:.2f}")
        self.get_logger().info(f"Min reward: {min_reward:.2f}")
        self.get_logger().info(f"Max reward: {max_reward:.2f}")
        
        # Save results
        if self.record_data:
            results = {
                'algorithm': self.algorithm,
                'model_path': self.model_path,
                'num_episodes': self.num_episodes,
                'rewards': self.all_rewards,
                'avg_reward': avg_reward,
                'std_reward': std_reward,
                'min_reward': min_reward,
                'max_reward': max_reward
            }
            
            import json
            with open(os.path.join(self.data_dir, 'results.json'), 'w') as f:
                json.dump(results, f, indent=4)
            
            self.get_logger().info(f"Saved results to {os.path.join(self.data_dir, 'results.json')}")
        
        # Stop timer
        self.timer.cancel()


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Evaluate RL agent for F1TENTH')
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to trained model')
    parser.add_argument('--algorithm', type=str, default='dqn', choices=['dqn', 'ppo'],
                        help='RL algorithm to use (dqn or ppo)')
    parser.add_argument('--num-episodes', type=int, default=10,
                        help='Number of episodes to evaluate')
    parser.add_argument('--record-data', action='store_true',
                        help='Record trajectory data')
    parser.add_argument('--visualization', action='store_true',
                        help='Enable visualization (not implemented yet)')
    
    args = parser.parse_args()
    
    # Initialize ROS
    rclpy.init()
    
    # Create and run evaluation node
    node = EvaluationNode(args)
    
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
