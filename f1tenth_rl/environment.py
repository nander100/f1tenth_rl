#!/usr/bin/env python3

import numpy as np
import math
from geometry_msgs.msg import Pose, Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseWithCovarianceStamped
import time

class F1TenthEnv:
    """
    Environment class that handles interaction with the F1TENTH simulator
    """
    def __init__(self, node=None):
        # Track dimensions (approximate values for typical F1TENTH tracks)
        self.track_width = 2.0  # meters
        
        # Previous speed and pose for calculating differences
        self.prev_speed = 0.0
        self.prev_pose = None
        
        # Track progress tracking
        self.start_position = None
        self.lap_progress = 0.0
        self.lap_count = 0
        
        # Collision detection
        self.collision_threshold = 0.3  # meters
        
        # Episode tracking
        self.episode_step = 0
        self.max_episode_steps = 1000

        # Store node reference for reset functionality
        self.node = node
        
        # Store the starting position
        self.start_position = [0.0, 0.0, 0.0]  # [x, y, theta]
        
        # Create reset publisher if node is provided
        if self.node is not None:
            from geometry_msgs.msg import PoseWithCovarianceStamped
            self.reset_pub = self.node.create_publisher(
                PoseWithCovarianceStamped,
                '/initialpose',
                10
            )

    def reset_car_position(self):
        """Reset the car to its starting position"""
        if not hasattr(self, 'reset_pub'):
            return False
            
        # Create the pose message
        pose_msg = PoseWithCovarianceStamped()
        pose_msg.header.frame_id = 'map'
        pose_msg.header.stamp.sec = int(time.time())
        
        # Set position (x, y, z)
        pose_msg.pose.pose.position.x = self.start_position[0]
        pose_msg.pose.pose.position.y = self.start_position[1]
        pose_msg.pose.pose.position.z = 0.0
        
        # Set orientation (as quaternion from yaw)
        from tf_transformations import quaternion_from_euler
        q = quaternion_from_euler(0, 0, self.start_position[2])
        pose_msg.pose.pose.orientation.x = q[0]
        pose_msg.pose.pose.orientation.y = q[1]
        pose_msg.pose.pose.orientation.z = q[2]
        pose_msg.pose.pose.orientation.w = q[3]
        
        # Publish the pose
        self.reset_pub.publish(pose_msg)
        
        # Sleep a bit to let the simulator respond
        time.sleep(0.5)
        
        return True
    
    def reset(self):
        """Reset environment state at the beginning of an episode"""
        self.prev_speed = 0.0
        self.prev_pose = None
        self.start_position = None
        self.lap_progress = 0.0
        self.lap_count = 0
        self.episode_step = 0
        
        return None  # In ROS we get states from callbacks, not directly
    
    def step(self, action, scan, odom):
        """
        Execute a step in the environment
        
        Args:
            action: [steering_angle, velocity]
            scan: LaserScan message
            odom: Odometry message
            
        Returns:
            next_state: current laser scan ranges
            reward: calculated reward
            done: whether episode is done
            info: additional information
        """
        # Extract current speed and position
        current_speed = self._get_speed_from_odom(odom)
        current_pose = odom.pose.pose
        
        # Initialize starting position if not set
        if self.start_position is None:
            self.start_position = current_pose
        
        # Check for collision
        collision = self._check_collision(scan)
        
        # Calculate reward components
        speed_reward = self._speed_reward(current_speed)
        steering_penalty = self._steering_penalty(action[0])
        progress_reward = self._progress_reward(current_pose)
        collision_penalty = -100.0 if collision else 0.0
        
        # Combine reward components
        reward = speed_reward + steering_penalty + progress_reward + collision_penalty
        
        # Check if episode is done
        self.episode_step += 1
        done = collision or self.episode_step >= self.max_episode_steps
        
        # Extract state from laser scan
        ranges = np.array(scan.ranges)
        ranges[np.isinf(ranges)] = 10.0  # Replace inf with large value
        state = ranges / 10.0  # Normalize to [0, 1]
        
        # Additional info
        info = {
            'speed': current_speed,
            'collision': collision,
            'lap_progress': self.lap_progress,
            'lap_count': self.lap_count
        }
        
        # Update previous values
        self.prev_speed = current_speed
        self.prev_pose = current_pose
        
        return state, reward, done, info
    
    def _get_speed_from_odom(self, odom):
        """Extract linear speed from odometry message"""
        if odom is None:
            return 0.0
            
        vx = odom.twist.twist.linear.x
        vy = odom.twist.twist.linear.y
        return math.sqrt(vx**2 + vy**2)
    
    def _check_collision(self, scan):
        """Check if any laser scan ray indicates a collision"""
        if scan is None:
            return False
            
        # Check if any laser reading is below threshold
        ranges = np.array(scan.ranges)
        return np.any(ranges < self.collision_threshold)
    
    def _speed_reward(self, speed):
        """Reward for maintaining speed"""
        # Encourage the car to drive at a target speed (e.g., 2.0 m/s)
        target_speed = 2.0
        
        # Gaussian reward - highest at target speed, falls off as we deviate
        return 0.5 * math.exp(-0.5 * ((speed - target_speed) / 1.0)**2)
    
    def _steering_penalty(self, steering_angle):
        """Penalty for excessive steering"""
        # Discourage sharp turns
        return -0.5 * abs(steering_angle)
    
    def _progress_reward(self, current_pose):
        """Reward for making progress around the track"""
        if self.prev_pose is None:
            return 0.0
            
        # Calculate distance traveled
        dx = current_pose.position.x - self.prev_pose.position.x
        dy = current_pose.position.y - self.prev_pose.position.y
        distance = math.sqrt(dx**2 + dy**2)
        
        # Simple progress reward based on distance
        # In a more advanced implementation, you would track progress along the track centerline
        return distance
