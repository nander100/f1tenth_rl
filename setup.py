from setuptools import setup
import os
from glob import glob

package_name = 'f1tenth_rl'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name, 
              package_name + '.models',
              package_name + '.utils'],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools',
                      'numpy',
                      'torch',
                      'gym',
                      'matplotlib'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='your_email@example.com',
    description='ROS2 package for reinforcement learning with F1TENTH simulator',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'rl_agent = f1tenth_rl.rl_agent_node:main',
            'train_rl_agent = scripts.train_rl_agent:main',
            'eval_rl_agent = scripts.eval_rl_agent:main',
        ],
    },
)
