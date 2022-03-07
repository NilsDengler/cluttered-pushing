#!/usr/bin/env python3
import os
import sys
sys.path.append(os.getcwd()+'/build')
import LazyThetaStarPython
import time
import h5py
import numpy as np
import matplotlib.pyplot as plt
import random
import cv2

def plot_path(orig, path, x=None):
    plt_fig = plt.figure()  # figsize=(1, 2))

    plt.subplot(1, 2, 1)
    plt.imshow(path)
    plt.axis("off")
    plt.show(block=False)

    # Plot reconstruction
    plt.subplot(1, 2, 2)
    plt.imshow(orig)
    plt.axis("off")
    plt.show(block=False)

    if x is not None:
       plt.subplot(1, 3, 3)
       plt.imshow(x)
       plt.axis("off")
       plt.show(block=False)

    plt.show()


def floodfill(matrix, p):
    matrix = matrix.astype(np.uint8)
    height, width = matrix.shape[:2]
    mask = np.zeros((height + 2, width + 2), np.uint8)
    cv2.floodFill(matrix, mask, (p[0],p[1]), 0)
    return matrix

def h5_to_np(h5_data, np_data):
    for i in range(len(h5_data)):
        dset = h5_data["data_" + str(i)]
        np_data[i] = np.asarray(dset[:])
    return np_data

if __name__ == "__main__":
    # define the world map
    map_width = 256
    map_height = 256
    depth_path = "/home/user/dengler/Documents/robot_arm_RL/pushing_net/push_gym/push_gym/astar_data/depth_data_23.h5"
    coord_path = "/home/user/dengler/Documents/robot_arm_RL/pushing_net/push_gym/push_gym/astar_data/astar_data_23.h5"
    astar_path = "/home/user/dengler/Documents/robot_arm_RL/pushing_net/push_gym/push_gym/astar_data/coord_data_23.h5"
    h5f_depth = h5py.File(depth_path, "r")
    h5f_astar = h5py.File(astar_path, "r")
    h5f_coord = h5py.File(coord_path, "r")
    depth_data = np.zeros((len(h5f_depth), 256, 256), dtype=np.uint8)
    astar_data = np.zeros((len(h5f_astar), 256, 256), dtype=np.uint8)
    coord_data = np.zeros((len(h5f_coord), 4), dtype=np.uint8)
    depth_data = h5_to_np(h5f_depth, depth_data)
    astar_data = h5_to_np(h5f_astar, astar_data)
    coord_data = h5_to_np(h5f_coord, coord_data)
    complete_start_time = time.time()
    random_int = random.randint(0, 22)
    print(random_int)
    start = list(reversed((coord_data[random_int][1], coord_data[random_int][0])))
    end = list(reversed((coord_data[random_int][3], coord_data[random_int][2])))
    print(start, end)
    t0 = time.time()
    depth_orig = np.copy(depth_data[random_int])
    # cut arm from image and set it to 0 and 1
    depth_wo_arm = np.copy(depth_data[random_int])
    depth_wo_arm[depth_wo_arm < 219] = 255
    depth_wo_arm[depth_wo_arm < 220] = 1
    depth_wo_arm[depth_wo_arm >= 220] = 0
    # cut object from image and dilate objects
    kernel_dilate = (np.ones((5, 5), np.float32)) / 25
    kernel_erode = (np.ones((3, 3), np.float32)) / 9
    depth_only_obstacles = np.copy(depth_wo_arm)
    #depth_only_obstacles = floodfill(depth_only_obstacles, start)
    depth_only_obstacles = cv2.erode(depth_only_obstacles, kernel_erode, iterations=1)
    depth_only_obstacles = cv2.dilate(depth_only_obstacles, kernel_dilate, iterations=2)

    depth_only_obstacles[depth_only_obstacles==1] = 255
    world_map = depth_only_obstacles.flatten().tolist()
    # This is for a single start and goal
    #start = [coord_data[random_int][1], coord_data[random_int][0]]
    #end = [coord_data[random_int][3], coord_data[random_int][2]]
    t1 = time.time()
    # solve it
    path, distance = LazyThetaStarPython.FindPath(start, end, world_map, map_width, map_height)
    t2 = time.time()
    print("This is the path. Time used [sec]:" + str(t2 - t1))
    print("total time for programm: " + str(t2 - t0))
    print("Total distance: " + str(distance))
    path_vis = cv2.cvtColor(depth_only_obstacles.copy(), cv2.COLOR_GRAY2RGB)
    for idx in range(0, len(path)-2, 2):
        path_vis = cv2.line(path_vis, (path[idx], path[idx+1]), (path[idx+2], path[idx+3]), (0, 255, 0), thickness=2)
    plot_path(depth_orig, path_vis)
