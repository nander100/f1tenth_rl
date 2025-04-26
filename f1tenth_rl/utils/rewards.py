#!/usr/bin/env python3

import numpy as np
import math
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Pose
from f1tenth_rl.utils.farthest_point import calculate_farthest_point_steering

class RewardCalculator:
    """
    Class responsible for calculating rewards for F1TENTH RL agents.
    Consolidates all reward functions into a single class.
    """
    def __init__(self, 
                 collision_penalty=-500.0,
                 speed_reward_weight=0.3,
                 steering_penalty_weight=-10.0,
                 progress_reward_weight=5.0,
                 centerline_reward_weight=25.0,
                 farthest_point_weight=1.0,
                 track_width=2.0,
                 collision_threshold=0.3):
        # Store configuration parameters
        self.collision_penalty = collision_penalty
        self.speed_reward_weight = speed_reward_weight
        self.steering_penalty_weight = steering_penalty_weight
        self.progress_reward_weight = progress_reward_weight
        self.centerline_reward_weight = centerline_reward_weight
        self.farthest_point_weight = farthest_point_weight
        
        # Environmental parameters
        self.track_width = track_width
        self.collision_threshold = collision_threshold
        
        # State tracking
        self.prev_speed = 0.0
        self.prev_pose = None
        self.lap_progress = 0.0

    def calculate_reward(self, scan, odom, prev_odom, action=None, use_farthest_point=True):
        """
        Calculate comprehensive reward for the current state and action
        
        Args:
            scan: Current laser scan
            odom: Current odometry
            prev_odom: Previous odometry
            action: Action taken (optional, [steering_angle, velocity])
            use_farthest_point: Whether to include farthest point reward
            
        Returns:
            reward: Calculated reward value
            done: Whether the episode is done
            info: Dictionary with reward component breakdown
        """
        # Initialize reward and info dictionary
        reward = 0.0
        done = False
        info = {}
        
        # 1. Collision detection and penalty
        min_distance = self._get_min_distance(scan)
        collision = min_distance < self.collision_threshold
        
        if collision:
            reward += self.collision_penalty
            done = True
            info['collision_penalty'] = self.collision_penalty
            # Early return on collision
            return reward, done, info
        
        # 2. Speed reward
        current_speed = self._get_speed_from_odom(odom)
        speed_reward = self._calculate_speed_reward(current_speed)
        reward += self.speed_reward_weight * speed_reward
        info['speed_reward'] = self.speed_reward_weight * speed_reward
        
        # 3. Steering penalty (if action is provided)
        if action is not None:
            steering_angle = action[0]
            steering_penalty = self._calculate_steering_penalty(steering_angle)
            reward += self.steering_penalty_weight * steering_penalty
            info['steering_penalty'] = self.steering_penalty_weight * steering_penalty
        
        # 4. Progress reward
        current_pose = odom.pose.pose
        if prev_odom is not None:
            prev_pose = prev_odom.pose.pose
            progress_reward = self._calculate_progress_reward(current_pose, prev_pose)
            reward += self.progress_reward_weight * progress_reward
            info['progress_reward'] = self.progress_reward_weight * progress_reward
        
        # 5. Centerline following reward
        centerline_reward = self._calculate_centerline_reward(scan)
        reward += self.centerline_reward_weight * centerline_reward
        info['centerline_reward'] = self.centerline_reward_weight * centerline_reward
        
        # 6. Farthest point steering reward (if enabled)
        if use_farthest_point and self.farthest_point_weight > 0:
            # Extract current steering angle
            if action is not None:
                current_steering = action[0]
            else:
                # Estimate from pose if action not provided
                current_steering = self._extract_steering_from_pose(current_pose)
                
            # Calculate farthest point steering suggestion
            farthest_steering, farthest_distance = calculate_farthest_point_steering(scan, noise_level=0.0)
            farthest_reward = self._calculate_farthest_point_reward(current_steering, farthest_steering)
            reward += self.farthest_point_weight * farthest_reward
            info['farthest_point_reward'] = self.farthest_point_weight * farthest_reward
            info['farthest_steering'] = farthest_steering
            info['farthest_distance'] = farthest_distance
        
        # Update state for next calculation
        self.prev_speed = current_speed
        self.prev_pose = current_pose
        
        return reward, done, info

    def _get_min_distance(self, scan):
        """Get minimum distance from laser scan"""
        if scan is None:
            return 10.0  # Return large value if no scan available
            
        ranges = np.array(scan.ranges)
        ranges[np.isinf(ranges)] = 10.0  # Replace inf with large value
        return np.min(ranges)

    def _get_speed_from_odom(self, odom):
        """Extract linear speed from odometry message"""
        if odom is None:
            return 0.0
            
        vx = odom.twist.twist.linear.x
        vy = odom.twist.twist.linear.y
        return math.sqrt(vx**2 + vy**2)

    def _calculate_speed_reward(self, speed, target_speed=2.0):
        """Reward for maintaining target speed"""
        # Gaussian reward - highest at target speed, falls off as we deviate
        # The 0.5 multiplier and 1.0 divisor control the width of the Gaussian
        return math.exp(-0.5 * ((speed - target_speed) / 1.0)**2)

    def _calculate_steering_penalty(self, steering_angle):
        """Penalty for excessive steering"""
        # Normalized penalty based on steering angle magnitude
        return -abs(steering_angle)

    def _calculate_progress_reward(self, current_pose, prev_pose):
        """Reward for making progress around the track"""
        # Calculate distance traveled
        dx = current_pose.position.x - prev_pose.position.x
        dy = current_pose.position.y - prev_pose.position.y
        distance = math.sqrt(dx**2 + dy**2)
        
        # Simple progress reward based on distance
        # In a more advanced implementation, you would track progress along the track centerline
        return distance

    def _calculate_centerline_reward(self, scan):
        """Reward for staying near the center of the track"""
        # Estimate distance to centerline using laser scan
        # Assuming forward-facing lidar with symmetric field of view
        
        # Get number of scan points
        num_points = len(scan.ranges)
        
        # Calculate indices for left and right quarters
        left_quarter = num_points // 4
        right_quarter = 3 * num_points // 4
        
        # Get points around left and right quarter
        left_dists = scan.ranges[left_quarter-5:left_quarter+5]  # 10 points around left quarter
        right_dists = scan.ranges[right_quarter-5:right_quarter+5]  # 10 points around right quarter

        # Filter out inf values
        left_dists = [d for d in left_dists if not math.isinf(d)]
        right_dists = [d for d in right_dists if not math.isinf(d)]

        # Get average distances (if lists aren't empty)
        left_dist = sum(left_dists) / len(left_dists) if left_dists else 10.0
        right_dist = sum(right_dists) / len(right_dists) if right_dists else 10.0
            
        # Perfect centerline following would have equal distances on both sides
        centerline_error = abs(left_dist - right_dist)
        
        # Exponential reward - highest when error is zero, falls off as error increases
        return math.exp(-6.0 * centerline_error)

    def _calculate_farthest_point_reward(self, current_steering, farthest_steering):
        """Reward for steering towards the farthest point"""
        # Calculate the difference between current steering and farthest point steering
        steering_diff = abs(current_steering - farthest_steering)
        
        # Gaussian reward - highest when steering matches farthest point direction
        # Lower as steering deviates
        return math.exp(-5.0 * steering_diff**2)

    def _extract_steering_from_pose(self, pose):
        """Extract approximate steering angle from pose orientation"""
        # Calculate yaw from quaternion
        qx = pose.orientation.x
        qy = pose.orientation.y
        qz = pose.orientation.z
        qw = pose.orientation.w
        
        # Calculate yaw from quaternion (simplified to just get the steering direction)
        steering_angle = 2.0 * math.atan2(
            2.0 * (qw * qz + qx * qy),
            1.0 - 2.0 * (qy**2 + qz**2)
        )
        
        return steering_angle

# Backward compatibility function (will use the RewardCalculator class internally)
def calculate_reward(scan, odom, prev_odom, use_farthest_point=True, farthest_point_weight=0.5):
    """
    Legacy interface for backward compatibility.
    Uses the RewardCalculator class internally.
    """
    # Create a reward calculator with default parameters
    calculator = RewardCalculator(
        collision_penalty=-100.0,  # Same as in original function
        speed_reward_weight=1.0,   # Adjusted to match original scale
        steering_penalty_weight=-0.2,  # Same as in original function
        progress_reward_weight=5.0,  # Same as in original function
        centerline_reward_weight=1.0,  # Adjusted to match original scale
        farthest_point_weight=farthest_point_weight
    )
    
    # Calculate reward using the calculator
    reward, done, _ = calculator.calculate_reward(scan, odom, prev_odom, use_farthest_point=use_farthest_point)
    
    return reward, done
