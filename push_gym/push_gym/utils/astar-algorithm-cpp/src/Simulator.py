#!/usr/bin/env python3
import os
import sys
sys.path.append(os.getcwd()+'/src')
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from random import randint


class Simulator(object):
    resolution: int
    map_width: int
    map_height: int
    value_non_obs: int
    value_obs: int
    size_obs_width: int
    size_obs_height: int

    def __init__(self, 
        map_width_meter: float, 
        map_height_meter: float, 
        resolution: int,
        value_non_obs: int,
        value_obs: int):
        """
        Constructor

        The cell is empty if value_non_obs, the cell is blocked if value_obs.
        """

        # map resolution, how many cells per meter
        self.resolution = resolution
        # how many cells for width and height
        map_width = map_width_meter * resolution
        map_height = map_height_meter * resolution

        # check if these are integers
        assert (map_width.is_integer() == True)
        assert (map_height.is_integer() == True)

        self.map_width = int(map_width)
        self.map_height = int(map_height)

        self.value_non_obs = value_non_obs
        self.value_obs = value_obs

        # create an empty map
        self.map_array = self.value_non_obs * np.ones((self.map_height, self.map_width)).astype(int)

    def generate_random_obs(self, num_obs: int, size_obs: list):
        self.size_obs_width = round(size_obs[0] * self.resolution)
        self.size_obs_height = round(size_obs[1] * self.resolution)
        if num_obs > 0:
            for idx_obs in range(0, num_obs):
                # [width, height]
                obs_left_bottom_corner = [randint(1,self.map_width-self.size_obs_width-1), randint(1, self.map_height-self.size_obs_height-1)]

                obs_mat = self.map_array[obs_left_bottom_corner[1]:obs_left_bottom_corner[1]+self.size_obs_height][:, obs_left_bottom_corner[0]:obs_left_bottom_corner[0]+self.size_obs_width]

                self.map_array[obs_left_bottom_corner[1]:obs_left_bottom_corner[1]+self.size_obs_height][:, obs_left_bottom_corner[0]:obs_left_bottom_corner[0]+self.size_obs_width] \
                    = self.value_obs * np.ones(obs_mat.shape)

    def plot_single_path(self, *arguments):
        """
        Simulator.visualize(path) # plot a path
        Simulator.visualize(path_full, path_short) # plot two paths

        path is a list for the trajectory. [x[0], y[0], x[1], y[1], ...]
        """
        if len(arguments[0]) > 0:
            fig_map, ax_map = plt.subplots(1, 1)
            
            cmap = matplotlib.colors.ListedColormap(['white','black'])
            ax_map.pcolor(self.map_array, cmap=cmap, edgecolors='k')

            ax_map.scatter(arguments[0][0]+0.5, arguments[0][1]+0.5, label="start")
            ax_map.scatter(arguments[0][-2]+0.5, arguments[0][-1]+0.5, label="goal")
            ax_map.plot(list(map(lambda x:x+0.5, arguments[0][0::2])),
                        list(map(lambda x:x+0.5, arguments[0][1::2])), label="path")
            if len(arguments) == 2:
                ax_map.plot(list(map(lambda x:x+0.5, arguments[1][0::2])),
                    list(map(lambda x:x+0.5, arguments[1][1::2])), label="path")
            ax_map.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax_map.set_xlabel("x")
            ax_map.set_ylabel("y")
            ax_map.set_aspect('equal')
            ax_map.set_xlim([0, self.map_width])
            ax_map.set_ylim([0, self.map_height])
            plt.show(block=False)
        else:
            print("A Star didn't find a path!")

    def plot_many_path(self, path_many: list, agent_position: list, targets_position: list):
        """
        path_many is a list for the trajectory. [[x[0],y[0],x[1],y[1], ...], [x[0],y[0],x[1],y[1], ...], ...]
        """
        fig_map, ax_map = plt.subplots(1, 1)

        cmap = matplotlib.colors.ListedColormap(['white','black'])
        ax_map.pcolor(self.map_array, cmap=cmap, edgecolors='k')

        ax_map.scatter(agent_position[0]+0.5, agent_position[1]+0.5, label="start")
        for idx_target in range(0, int(len(targets_position)/2)):
            ax_map.scatter(targets_position[2*idx_target]+0.5, targets_position[2*idx_target+1]+0.5, label="goal")

        for idx_path in range(0, len(path_many)):
            ax_map.plot(list(map(lambda x:x+0.5, path_many[idx_path][0::2])),
                list(map(lambda x:x+0.5, path_many[idx_path][1::2])))

        ax_map.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax_map.set_xlabel("x")
        ax_map.set_ylabel("y")
        ax_map.set_aspect('equal')
        ax_map.set_xlim([0, self.map_width])
        ax_map.set_ylim([0, self.map_height])
        plt.show(block=False)

    def generate_start_and_goals(self, num_targets: int):
        start = [randint(5,self.map_width-1), randint(5,self.map_height-1)]
        while self.map_array[start[1]][start[0]] != self.value_non_obs:
            start = [randint(5,self.map_width-1), randint(5,self.map_height-1)]
            # print("Start is inside an obstacle. Re-generate a new start.")
        targets = list()
        for idx in range(0, num_targets):
            goal = [randint(20,self.map_width-1), randint(20,self.map_height-1)]
            while self.map_array[goal[1]][goal[0]] != self.value_non_obs:
                goal = [randint(20,self.map_width-1), randint(20,self.map_height-1)]
                # print("Target is inside an obstacle. Re-generate a new target.")
            targets.extend(goal)
        return start, targets
