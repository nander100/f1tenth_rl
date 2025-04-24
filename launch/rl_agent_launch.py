from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # Path to the parameter file
    config_file = os.path.join(
        get_package_share_directory('f1tenth_rl'),
        'config',
        'agent_params.yaml'
    )
    
    # Launch arguments
    training_mode_arg = DeclareLaunchArgument(
        'training_mode',
        default_value='true',
        description='Enable training mode (true) or testing mode (false)'
    )
    
    model_type_arg = DeclareLaunchArgument(
        'model_type',
        default_value='dqn',
        description='RL algorithm to use (dqn, ppo, etc.)'
    )
    
    model_path_arg = DeclareLaunchArgument(
        'model_path',
        default_value='',
        description='Path to pretrained model (if not training from scratch)'
    )
    
    save_path_arg = DeclareLaunchArgument(
        'save_path',
        default_value='models/',
        description='Directory to save trained models'
    )
    
    # RL agent node
    rl_agent_node = Node(
        package='f1tenth_rl',
        executable='rl_agent',
        name='rl_agent',
        parameters=[
            config_file,
            {
                'training_mode': LaunchConfiguration('training_mode'),
                'model_type': LaunchConfiguration('model_type'),
                'model_path': LaunchConfiguration('model_path'),
                'save_path': LaunchConfiguration('save_path')
            }
        ],
        output='screen'
    )
    
    return LaunchDescription([
        training_mode_arg,
        model_type_arg,
        model_path_arg,
        save_path_arg,
        rl_agent_node
    ])
