#!/usr/bin/env python3
import os
import sys
sys.path.append(os.getcwd()+'/build')
import AStarPython


if __name__ == "__main__":

    # define the world map
    map_width = 20
    map_height = 20

    world_map = [
    # 00 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19
        1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,   # 00
        1,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,1,   # 01
        1,9,9,1,1,9,9,9,1,9,1,9,1,9,1,9,9,9,1,1,   # 02
        1,9,9,1,1,9,9,9,1,9,1,9,1,9,1,9,9,9,1,1,   # 03
        1,9,1,1,1,1,9,9,1,9,1,9,1,1,1,1,9,9,1,1,   # 04
        1,9,1,1,9,1,1,1,1,9,1,1,1,1,9,1,1,1,1,1,   # 05
        1,9,9,9,9,1,1,1,1,1,1,9,9,9,9,1,1,1,1,1,   # 06
        1,9,9,9,9,9,9,9,9,1,1,1,9,9,9,9,9,9,9,1,   # 07
        1,9,1,1,1,1,1,1,1,1,1,9,1,1,1,1,1,1,1,1,   # 08
        1,9,1,9,9,9,9,9,9,9,1,1,9,9,9,9,9,9,9,1,   # 09
        1,9,1,1,1,1,9,1,1,9,1,1,1,1,1,1,1,1,1,1,   # 10
        1,9,9,9,9,9,1,9,1,9,1,9,9,9,9,9,1,1,1,1,   # 11
        1,9,1,9,1,9,9,9,1,9,1,9,1,9,1,9,9,9,1,1,   # 12
        1,9,1,9,1,9,9,9,1,9,1,9,1,9,1,9,9,9,1,1,   # 13
        1,9,1,1,1,1,9,9,1,9,1,9,1,1,1,1,9,9,1,1,   # 14
        1,9,1,1,9,1,1,1,1,9,1,1,1,1,9,1,1,1,1,1,   # 15
        1,9,9,9,9,1,1,1,1,1,1,9,9,9,9,1,1,1,1,1,   # 16
        1,1,9,9,9,9,9,9,9,1,1,1,9,9,9,1,9,9,9,9,   # 17
        1,9,1,1,1,1,1,1,1,1,1,9,1,1,1,1,1,1,1,1,   # 18
        1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1    # 19
    ]

    # for AStarPython, 0 for no obstacles; 255 for obstacles
    for idx in range(len(world_map)):
        if world_map[idx] == 9:
            world_map[idx] = 255
        else:
            world_map[idx] = 0

    # define the start and goal
    start = [0, 0]
    end = [14, 10]
    path, steps_used = AStarPython.FindPath(start, end, world_map, map_width, map_height)
    print("This is the path. " + "Steps used:" + str(steps_used))
    for idx in range(0,len(path),2):
        str_print = str(path[idx]) + ', ' + str(path[idx+1])
        print(str_print)

    # This is for an agent and a set of targets
    agent_position = [0, 0]
    targets_position = [0,19, 19,19, 19,0]
    path_many, steps_all = AStarPython.FindPathAll(agent_position, targets_position, world_map, map_width, map_height)
    print("These are all the paths:")
    for i in range(0,len(path_many),1):
        print("This is a path. " + "Steps used:" + str(steps_all[i]))
        for j in range(0,len(path_many[i]),2):
            str_print = str(path_many[i][j]) + ', ' + str(path_many[i][j+1])
            print(str_print)