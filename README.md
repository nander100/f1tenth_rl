# F1TENTH Reinforcement Learning Package

A ROS2 package for training and deploying reinforcement learning agents in the F1TENTH simulator.

## Overview

This package provides a framework for implementing reinforcement learning agents for autonomous racing with the F1TENTH simulator. It includes:

- A ROS2 node that interfaces with the F1TENTH simulator
- A reinforcement learning environment wrapper
- DQN (Deep Q-Network) implementation
- Customizable reward functions
- Tools for training and evaluation

## Installation
See setup.md

### Training an RL Agent

1. Start the F1TENTH simulator:
```bash
colcon build # build the workspsace
source /opt/ros/foxy/setup.bash
source install/local_setup.bash
ros2 launch f1tenth_gym_ros gym_bridge_launch.py
```

2. In a new terminal, launch the RL agent in training mode:
```bash
source /opt/ros/foxy/setup.bash
source install/local_setup.bash
ros2 launch f1tenth_rl rl_agent_launch.py training_mode:=true model_type:=dqn
```

The agent will begin training and save model checkpoints periodically to the specified `save_path` (default: `models/`).

### Deploying a Trained Agent

1. Start the F1TENTH simulator as described above.

2. Launch the RL agent in deployment mode, providing the path to your trained model:
```bash
source /opt/ros/foxy/setup.bash
source install/local_setup.bash
ros2 launch f1tenth_rl rl_agent_launch.py training_mode:=false model_path:=/path/to/your/model.pt
```

## Configuration

You can modify the parameters in `config/agent_params.yaml` to adjust:
- Training hyperparameters (learning rate, batch size, etc.)
- Reward function components and weights

## Implementation Details

### RL Agent Node

The `rl_agent_node.py` implements the ROS2 node that:
- Subscribes to laser scan and odometry data from the simulator
- Processes observations and calculates rewards
- Trains the reinforcement learning model
- Publishes drive commands to control the vehicle

### Environment Interface

The `environment.py` file provides a wrapper around the simulator that:
- Converts ROS messages into state representations
- Calculates rewards based on driving performance
- Detects episode termination conditions
- Tracks episode progress

### DQN Implementation

The `models/dqn.py` file implements the Deep Q-Network algorithm with:
- Neural network architecture for Q-function approximation
- Experience replay for stable learning
- Target network for reducing overestimation
- Epsilon-greedy exploration strategy

### Reward Function

The `utils/rewards.py` file defines the reward function components:
- Speed rewards for maintaining target velocity
- Progress rewards for lap completion
- Penalties for collisions and excessive steering
- Rewards for centerline following

## Extending the Package

### Adding New RL Algorithms

To implement a new RL algorithm (e.g., PPO):
1. Create a new file in the `models/` directory (e.g., `ppo.py`)
2. Implement the agent class with appropriate methods
3. Update the `rl_agent_node.py` to support the new algorithm
4. Update the parameter file with relevant hyperparameters

### Customizing Reward Functions

You can modify the reward function by:
1. Editing the `calculate_reward` function in `utils/rewards.py`
2. Adjusting reward component weights in the parameter file

## References

- F1TENTH Gym: https://github.com/f1tenth/f1tenth_gym
- F1TENTH Gym ROS Bridge: https://github.com/f1tenth/f1tenth_gym_ros
