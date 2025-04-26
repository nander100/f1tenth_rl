#!/usr/bin/env python3

import numpy as np
import math

def calculate_farthest_point_steering(scan, fov=270.0, noise_level=0.1):
    """
    Calculate steering angle towards the farthest point in front of the car.
    
    Args:
        scan: LaserScan message containing range data
        fov: Field of view of the LiDAR in degrees (default: 270.0)
        noise_level: Level of noise to add to the steering suggestion (default: 0.1)
        
    Returns:
        steering_suggestion: Recommended steering angle in radians
        distance: Distance to the farthest point
    """
    if scan is None:
        return 0.0, 0.0
    
    # Convert scan ranges to numpy array
    ranges = np.array(scan.ranges)
    
    # Replace inf values with a large but finite number
    ranges[np.isinf(ranges)] = 10.0
    
    # Calculate angle increment
    angle_increment = scan.angle_increment
    
    # Calculate the starting angle (typically -135 degrees for 270 degree FOV)
    start_angle = scan.angle_min
    
    # Define the front sector (e.g., -45 to 45 degrees)
    front_angle_range = 90.0  # degrees
    front_angle_rad = math.radians(front_angle_range / 2.0)
    
    # Calculate indices for the front sector
    front_start_idx = int((front_angle_rad + start_angle) / angle_increment)
    front_end_idx = int((-front_angle_rad + start_angle) / angle_increment)
    
    # Ensure the indices are within range
    if front_start_idx < 0:
        front_start_idx = 0
    if front_end_idx >= len(ranges):
        front_end_idx = len(ranges) - 1
    
    # If we need to wrap around (e.g., when front_start_idx > front_end_idx)
    if front_start_idx > front_end_idx:
        front_ranges = np.concatenate((ranges[front_start_idx:], ranges[:front_end_idx+1]))
        front_angles = np.concatenate((
            np.arange(front_start_idx, len(ranges)) * angle_increment + start_angle,
            np.arange(0, front_end_idx+1) * angle_increment + start_angle
        ))
    else:
        front_ranges = ranges[front_start_idx:front_end_idx+1]
        front_angles = np.arange(front_start_idx, front_end_idx+1) * angle_increment + start_angle
    
    # Find the index of the farthest point in the front sector
    if len(front_ranges) == 0:
        return 0.0, 0.0
        
    farthest_idx = np.argmax(front_ranges)
    farthest_distance = front_ranges[farthest_idx]
    farthest_angle = front_angles[farthest_idx]
    
    # Calculate steering suggestion (negative angle -> left, positive -> right)
    steering_suggestion = farthest_angle
    
    # Add noise to encourage exploration
    if noise_level > 0:
        noise = np.random.normal(0, noise_level)
        steering_suggestion += noise
        
        # Clip to reasonable steering range (-0.4 to 0.4 radians)
        steering_suggestion = np.clip(steering_suggestion, -0.4, 0.4)
    
    return steering_suggestion, farthest_distance
