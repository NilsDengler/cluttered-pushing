#!/usr/bin/env python3
import os
import sys
sys.path.append(os.getcwd()+'/build')
import AStarPython
import numpy as np
import matplotlib.pyplot as plt
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


def calculate_lazy_theta_star(depth_orig, start, end):
    map_width, map_height = depth_orig.shape[0], depth_orig.shape[1]
    # cut arm from image and set it to 0 and 1
    depth_wo_arm = np.copy(depth_orig)
    depth_wo_arm[depth_wo_arm < 219] = 255
    depth_wo_arm[depth_wo_arm < 220] = 1
    depth_wo_arm[depth_wo_arm >= 220] = 0
    # cut object from image and dilate objects
    kernel_dilate = (np.ones((7, 7), np.float32)) / 49
    kernel_erode = (np.ones((3, 3), np.float32)) / 9
    depth_only_obstacles = np.copy(depth_wo_arm)
    depth_only_obstacles = floodfill(depth_only_obstacles, start)
    depth_only_obstacles = cv2.erode(depth_only_obstacles, kernel_erode, iterations=1)
    depth_only_obstacles = cv2.dilate(depth_only_obstacles, kernel_dilate, iterations=2)
    depth_only_obstacles[depth_only_obstacles == 1] = 255
    world_map = depth_only_obstacles.flatten().tolist()
    #returns: path_short, path_smooth, steps_used
    return AStarPython.FindPath_test(start, end, world_map,map_width, map_height)

