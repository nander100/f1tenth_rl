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
pip install torch numpy matplotlib
colcon build --packages-select f1tenth_rl
```

4. Source and run sim
```bash
source /opt/ros/foxy/setup.bash
source install/local_setup.bash
ros2 launch f1tenth_gym_ros gym_bridge_launch.py
```
5.  Open a new terminal, run the docker image and run

```bash
docker exec -it <CONTAINER_ID> /bin/bash
ros2 launch f1tenth_rl rl_agent_launch.py training_mode:=true model_type:=dqn
ros2 launch f1tenth_rl rl_agent_launch.py training_mode:=true model_type:=dqn
```

