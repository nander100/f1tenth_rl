#!/usr/bin/env python3

import numpy as np
import math
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from f1tenth_rl.utils.farthest_point import calculate_farthest_point_steering

def calculate_reward(scan, odom, prev_odom, use_farthest_point=True, farthest_point_weight=0.5):
    """
    Calculate reward for the current state and action
    
    Args:
        scan: Current laser scan
        odom: Current odometry
        prev_odom: Previous odometry
        use_farthest_point: Whether to include farthest point reward
        farthest_point_weight: Weight for the farthest point steering component
        
    Returns:
        reward: Calculated reward value
        done: Whether the episode is done
    """
    # Initialize reward
    reward = 0.0
    done = False
    
    # 1. Collision penalty
    min_distance = min(scan.ranges)
    if min_distance < 0.3:  # Collision threshold
        reward -= 100.0
        done = True
        return reward, done
    
    # 2. Speed reward
    current_speed = _get_speed_from_odom(odom)
    target_speed = 2.0  # Target speed in m/s
    
    # Give higher reward for being close to target speed
    speed_reward = 1.0 * math.exp(-0.5 * ((current_speed - target_speed) / 1.0)**2)
    reward += speed_reward
    
    # 3. Distance to obstacles reward
    # Higher reward for keeping distance from obstacles
    distance_reward = 0.2 * min(min_distance, 1.0)  # Cap at 1.0 meter
    reward += distance_reward
    
    # 4. Progress reward
    # Calculate distance traveled
    current_pose = odom.pose.pose
    prev_pose = prev_odom.pose.pose
    
    dx = current_pose.position.x - prev_pose.position.x
    dy = current_pose.position.y - prev_pose.position.y
    distance_traveled = math.sqrt(dx**2 + dy**2)
    
    # Encourage forward movement
    progress_reward = 5.0 * distance_traveled
    reward += progress_reward
    
    # 5. Steering efficiency reward
    # Penalize excessive steering
    steering_angle = 2.0 * math.atan2(
        2.0 * (current_pose.orientation.w * current_pose.orientation.z + 
               current_pose.orientation.x * current_pose.orientation.y),
        1.0 - 2.0 * (current_pose.orientation.y**2 + current_pose.orientation.z**2)
    )
    
    # Calculate change in steering angle
    prev_steering_angle = 2.0 * math.atan2(
        2.0 * (prev_pose.orientation.w * prev_pose.orientation.z + 
               prev_pose.orientation.x * prev_pose.orientation.y),
        1.0 - 2.0 * (prev_pose.orientation.y**2 + prev_pose.orientation.z**2)
    )
    
    steering_change = abs(steering_angle - prev_steering_angle)
    steering_penalty = -0.2 * steering_change
    reward += steering_penalty
    
    # 6. Centerline following reward
    # Estimate distance to centerline using laser scan
    # Assuming forward-facing lidar with symmetric field of view
    left_quarter = len(scan.ranges) // 4
    right_quarter = 3 * len(scan.ranges) // 4
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
    centerline_reward = 1.0 * math.exp(-6.0 * centerline_error)  # Steeper falloff with error
    reward += centerline_reward
    
    # 7. Farthest point steering reward (if enabled)
    if use_farthest_point and farthest_point_weight > 0:
        # Calculate the "natural" steering direction towards the farthest point
        farthest_steering, _ = calculate_farthest_point_steering(scan, noise_level=0.0)
        
        # Extract current steering angle from the car
        current_steering = steering_angle
        
        # Reward for steering towards the farthest point (or close to it)
        # The closer the car's steering is to the farthest point direction, the higher the reward
        steering_diff = abs(current_steering - farthest_steering)
        
        # Gaussian reward - highest when steering matches farthest point direction
        # Lower as steering deviates
        farthest_reward = farthest_point_weight * math.exp(-5.0 * steering_diff**2)
        reward += farthest_reward
    
    return reward, done

def _get_speed_from_odom(odom):
    """Extract linear speed from odometry message"""
    vx = odom.twist.twist.linear.x
    vy = odom.twist.twist.linear.y
    return math.sqrt(vx**2 + vy**2)
