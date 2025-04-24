# F1TENTH RL Package Setup Guide

## Running in Docker

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
git clone https://github.com/nander100/f1tenth_rl.git
cd ..
colcon build --packages-select f1tenth_rl
```

4. Source and run sim
```bash
source /opt/ros/foxy/setup.bash
source install/local_setup.bash
ros2 launch f1tenth_gym_ros gym_bridge_launch.py
```
5.  Open a new terminal and run

```bash

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
