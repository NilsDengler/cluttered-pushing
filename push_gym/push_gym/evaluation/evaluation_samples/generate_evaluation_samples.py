import gym, push_gym
import sys



default_path = "/home/user/dengler/Documents/robot_arm_RL/pushing_net/"
sys.path.append(default_path + 'push_gym/push_gym/envs/')
sys.path.append(default_path + 'push_gym/push_gym/baseline_approach')
sys.path.append(default_path + 'push_gym/push_gym/')
sys.path.append(default_path + 'push_gym/push_gym/plot_scripts/')
sys.path.append(default_path + 'Networks/VAE/models/')
sys.path.append(default_path + 'utils/')
sys.path.append(default_path + 'push_gym/push_gym/rl_env/')
sys.path.append('/home/david/Arbeit/git/')
sys.path.append(default_path + 'utils/astar-algorithm-cpp/build')
sys.path.append(default_path + 'utils/Lazy-Theta-with-optimization-any-angle-pathfinding/build')

import json
# with open('./environment_sample_data.txt') as json_file:
#     data = json.load(json_file)
#     for p in data['samples']:
#         print(p["start"])


env = gym.make("EvaluationSampleEnvGUI-v0")
while True:
    pass
env.sample_data(1000)
