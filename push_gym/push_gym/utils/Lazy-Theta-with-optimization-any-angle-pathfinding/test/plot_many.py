#!/usr/bin/env python3
import os
import sys
sys.path.append(os.getcwd()+'/build')
sys.path.append(os.getcwd()+'/src')
import LazyThetaStarPython
import numpy as np
import time
from math import sqrt
import matplotlib.pyplot as plt
from Simulator import Simulator


if __name__ == "__main__":
    # define the world
    map_width_meter = 20.0
    map_height_meter = 20.0
    map_resolution = 2
    value_non_obs = 0 # the cell is empty
    value_obs = 255 # the cell is blocked
    # create a simulator
    MySimulator = Simulator(map_width_meter, map_height_meter, map_resolution, value_non_obs, value_obs)
    # number of obstacles
    num_obs = 50
    # [width, length] size of each obstacle [meter]
    size_obs = [1, 1]
    # generate random obstacles
    MySimulator.generate_random_obs(num_obs, size_obs)
    # convert 2D numpy array to 1D list
    world_map = MySimulator.map_array.flatten().tolist()

    num_run = 200
    max_num_targets = 51
    time_used_list = list()
    xticks_str_list = list()
    for num_targets in range (1,max_num_targets+1,10):
        time_used_single_list = list()
        for idx_run in range(0,num_run,1):
            # This is for an agent and a set of targets
            agent_position, targets_position = MySimulator.generate_start_and_goals(num_targets)

            t0 = time.time()
            # solve it
            path_many, distances_many = LazyThetaStarPython.FindPathMany(agent_position, targets_position, world_map, MySimulator.map_width, MySimulator.map_height)
            t1 = time.time()
            time_used = (t1 - t0) * 1000.0  # ms

            if len(distances_many) > 0:
                time_used_single_list.append(time_used)
            else:
                print("no solution!")

        time_used_list.append(time_used_single_list)
        xticks_str_list.append(str(num_targets))
        print(num_targets)

    xticks_list = range(1,len(time_used_list)+1,1)

    # create box plot
    fig1, ax1 = plt.subplots()
    ax1.set_title('Average solving time for 1 agent and multiple targets')
    # Creating plot
    bp = ax1.boxplot(time_used_list, showfliers=False)
    plt.xlabel('Number of targets')
    plt.ylabel('Solving time [ms]')
    plt.xticks(xticks_list, xticks_str_list)
    # show plot
    plt.show()
