#!/usr/bin/env python3
import os
import sys
sys.path.append(os.getcwd()+'/build/')
sys.path.append(os.getcwd()+'/src/')
print(sys.path)
import LazyThetaStarPython
import numpy as np
import time
from math import sqrt
import matplotlib.pyplot as plt
from Simulator import Simulator


if __name__ == "__main__":
    # define the world
    map_width_meter = 30.0
    map_height_meter = 20.0
    map_resolution = 2
    value_non_obs = 0 # the cell is empty
    value_obs = 255 # the cell is blocked
    # create a simulator
    MySimulator = Simulator(map_width_meter, map_height_meter, map_resolution, value_non_obs, value_obs)
    # number of obstacles
    num_obs = 250
    # [width, length] size of each obstacle [meter]
    size_obs = [0.5, 0.5]
    # generate random obstacles
    MySimulator.generate_random_obs(num_obs, size_obs)
    # convert 2D numpy array to 1D list
    world_map = MySimulator.map_array.flatten().tolist()

    # This is for a single start and goal
    num_targets = 1
    start, targets = MySimulator.generate_start_and_goals(num_targets)
    # solve it
    t0 = time.time()
    path_single, distance_single = LazyThetaStarPython.FindPath(start, targets, world_map, MySimulator.map_width, MySimulator.map_height)
    t1 = time.time()
    print("Time used for a single path is [sec]: " + str(t1-t0))
    print("This is the path.")
    print("Total distance: " + str(distance_single))
    distance_check = 0.0
    for idx in range(0,len(path_single),2):
        str_print = str(path_single[idx]) + ', ' + str(path_single[idx+1])
        print(str_print)
        if idx > 0:
            distance_check = distance_check + sqrt((path_single[idx]-path_single[idx-2])**2 + (path_single[idx+1]-path_single[idx-1])**2)
    print("Distance computed afterwards: " + str(distance_check))
    # visualization (uncomment next line if you want to visualize a single path)
    MySimulator.plot_single_path(path_single)

    # This is for an agent and a set of targets
    num_targets = 4
    agent_position, targets_position = MySimulator.generate_start_and_goals(num_targets)
    t0 = time.time()
    # solve it
    path_many, distances_many = LazyThetaStarPython.FindPathMany(agent_position, targets_position, world_map, MySimulator.map_width, MySimulator.map_height)
    t1 = time.time()
    print("These are all the paths. Time used [sec]:" + str(t1 - t0))
    for i in range(0,len(path_many),1):
        print("This is a path.")
        print("Total distance: " + str(distances_many[i]))
        distance_check = 0.0
        for j in range(0,len(path_many[i]),2):
            str_print = str(path_many[i][j]) + ', ' + str(path_many[i][j+1])
            print(str_print)
            if j > 0:
                distance_check = distance_check + sqrt((path_many[i][j]-path_many[i][j-2])**2 + (path_many[i][j+1]-path_many[i][j-1])**2)
        print("Distance computed afterwards: " + str(distance_check))

    # visualization (uncomment next line if you want to visualize a single path)
    MySimulator.plot_many_path(path_many, agent_position, targets_position)
    plt.show()
