## Prerequisites
pip3 install gym pybullet

## Install push_gym
cd ~/path/to/project/push_gym 
pip install -e .

## Usage
Create environment instance within your main project:  
```python
import gym, push_gym  
env = gym.make('NameOfYourEnvironment') # e.g. use 'pushing-v0'
```


