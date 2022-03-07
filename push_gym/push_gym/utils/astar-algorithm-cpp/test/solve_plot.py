#!/usr/bin/env python3
import os
import sys
sys.path.append(os.getcwd()+'/build')
sys.path.append(os.getcwd()+'/src')
import time
import AStarPython
import numpy as np
import matplotlib.pyplot as plt
from Simulator import Simulator


if __name__ == "__main__":
    # define the world
    map_width_meter = 20.0
    map_height_meter = 30.0
    map_resolution = 2
    value_non_obs = 0 # the cell is empty
    value_obs = 255 # the cell is blocked
    # create a simulator
    MySimulator = Simulator(map_width_meter, map_height_meter, map_resolution, value_non_obs, value_obs)
    # number of obstacles
    num_obs = 150
    # [width, length] size of each obstacle [meter]
    size_obs = [1, 1]
    # generate random obstacles
    MySimulator.generate_random_obs(num_obs, size_obs)
    # convert 2D numpy array to 1D list
    world_map = MySimulator.map_array.flatten().tolist()

    # define the start and goal
    num_targets = 1
    start, end = MySimulator.generate_start_and_goals(num_targets)
    # solve it
    t0 = time.time()
    path_short, steps_used = AStarPython.FindPath(start, end, world_map, MySimulator.map_width, MySimulator.map_height)
    t1 = time.time()
    print("Time used for a single path is [sec]:")
    print(t1-t0)
    print("This is the path. " + "Steps used:" + str(steps_used))
    for idx in range(0,len(path_short),2):
        str_print = str(path_short[idx]) + ', ' + str(path_short[idx+1])
        print(str_print)
    # visualization (uncomment next line if you want to visualize a single path)
    MySimulator.plot_single_path(path_short)

    # This is for an agent and a set of targets
    num_targets = 4
    agent_position, targets_position = MySimulator.generate_start_and_goals(num_targets)
    # solve it
    t0 = time.time()
    path_many, steps_all = AStarPython.FindPathAll(agent_position, targets_position, world_map, MySimulator.map_width, MySimulator.map_height)
    t1 = time.time()
    print("Time used for many paths is [sec]:")
    print(t1-t0)

    print("These are all the paths:")
    for i in range(0,len(path_many),1):
        print("This is a path. " + "Steps used:" + str(steps_all[i]))
        for j in range(0,len(path_many[i]),2):
            str_print = str(path_many[i][j]) + ', ' + str(path_many[i][j+1])
            print(str_print)
    # visualization
    MySimulator.plot_many_path(path_many, agent_position, targets_position)
    plt.show()
