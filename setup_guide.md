# F1TENTH RL Package Setup Guide


## Step 1: Install ROS2 Foxy

Follow the official ROS2 installation instructions: https://docs.ros.org/en/foxy/Installation.html

## Step 1: Go to /src

```bash
cd ~/f1tenth_ws/src
```

## Step 3: Install F1TENTH Gym

```bash
git clone https://github.com/f1tenth/f1tenth_gym
cd f1tenth_gym
pip3 install -e .
cd ..
```

## Step 4: Install F1TENTH Gym ROS Bridge

```bash
git clone https://github.com/f1tenth/f1tenth_gym_ros
```

## Step 5: Create the F1TENTH RL Package

```bash
# Create package directories
mkdir -p f1tenth_rl/f1tenth_rl/models
mkdir -p f1tenth_rl/f1tenth_rl/utils
mkdir -p f1tenth_rl/config
mkdir -p f1tenth_rl/launch
mkdir -p f1tenth_rl/resource/f1tenth_rl
touch f1tenth_rl/resource/f1tenth_rl/marker_file  # Empty file for resource indexing
```

## Step 6: Copy Package Files

Copy all the Python files from the artifacts to their respective locations:

1. `f1tenth_rl/f1tenth_rl/rl_agent_node.py`
2. `f1tenth_rl/f1tenth_rl/environment.py`
3. `f1tenth_rl/f1tenth_rl/models/dqn.py`
4. `f1tenth_rl/f1tenth_rl/models/ppo.py`
5. `f1tenth_rl/f1tenth_rl/utils/rewards.py`
6. `f1tenth_rl/launch/rl_agent_launch.py`
7. `f1tenth_rl/config/agent_params.yaml`
8. `f1tenth_rl/setup.py`
9. `f1tenth_rl/package.xml`

Don't forget to add `__init__.py` files:

```bash
touch f1tenth_rl/f1tenth_rl/__init__.py
touch f1tenth_rl/f1tenth_rl/models/__init__.py
touch f1tenth_rl/f1tenth_rl/utils/__init__.py
```

## Step 7: Install Dependencies

```bash
cd ~/f1tenth_ws
pip3 install torch numpy matplotlib
source /opt/ros/foxy/setup.bash
rosdep install -i --from-path src --rosdistro foxy -y
```

## Step 8: Build the Workspace

```bash
colcon build --packages-select f1tenth_rl
```

## Step 9: Running the Simulator and RL Agent

First, start the F1TENTH simulator:

```bash
source ~/f1tenth_ws/install/setup.bash
ros2 launch f1tenth_gym_ros gym_bridge_launch.py
```

In a new terminal, start the RL agent:

```bash
source ~/f1tenth_ws/install/setup.bash
ros2 launch f1tenth_rl rl_agent_launch.py
```

## Running in Docker

If you prefer to use Docker with the simulator:

1. Build the F1TENTH Gym ROS image first:
```bash
cd ~/f1tenth_ws/src/f1tenth_gym_ros
docker build -t f1tenth_gym_ros -f Dockerfile .
```

2. Start the container:
```bash
docker run -it \
  --gpus all \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v $(pwd):/sim_ws/src/f1tenth_gym_ros \
  --network=host \
  f1tenth_gym_ros
```

3. Then build your RL package inside the container:
```bash
cd /sim_ws/src
# Copy your package files here
cd ..
colcon build --packages-select f1tenth_rl
```

## Monitoring Training Progress

You can create monitoring scripts to track the training progress. A simple approach is to save episode rewards and plot them:

```python
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_rewards(rewards_file, save_path):
    rewards = np.loadtxt(rewards_file)
    plt.figure(figsize=(10, 6))
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Training Progress')
    plt.savefig(save_path)
    plt.close()

if __name__ == "__main__":
    plot_rewards('rewards.txt', 'training_progress.png')
```

## Troubleshooting

### Common Issues

1. **Display not working in Docker**: Make sure you've run `xhost +local:` before starting the container.

2. **Package not found**: Ensure you've sourced the workspace setup file:
```bash
source ~/f1tenth_ws/install/setup.bash
```

3. **Error importing PyTorch**: Double-check PyTorch installation:
```bash
pip3 install torch
```

4. **Simulator performance issues**: If using a less powerful computer, consider reducing the simulator's update rate in the F1TENTH Gym ROS config file.

### Getting Help

- F1TENTH Community: https://f1tenth.org/learn.html
- ROS2 Community: https://discourse.ros.org/
- Reinforcement Learning Resources: https://spinningup.openai.com/
