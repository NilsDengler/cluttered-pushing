#!/usr/bin/env python3
import os
import sys
import LazyThetaStarPython
import numpy as np
import matplotlib.pyplot as plt
import cv2

def floodfill(matrix, p):
    matrix = matrix.astype(np.uint8)
    height, width = matrix.shape[:2]
    mask = np.zeros((height + 2, width + 2), np.uint8)
    cv2.floodFill(matrix, mask, (p[0],p[1]), 0)
    return matrix

def save_path_images(img, path, current_steps):
    for idx in range(0, len(path) - 2, 2):
        img = cv2.line(img, (path[idx], path[idx + 1]), (path[idx + 2], path[idx + 3]), (0, 255, 0),thickness=2)
    plt.imsave("/home/user/dengler/test_path_saving/save_" + str(current_steps) + "_orig.png", img)

def get_path_img(img, path):
    if len(path) < 4: return img
    for idx in range(0, len(path) - 2, 2):
        img = cv2.line(img, (path[idx], path[idx + 1]), (path[idx + 2], path[idx + 3]), (0, 255, 0),thickness=1)
    return img

def create_np_path(p):
    return [[p[idx],p[idx+1]] for idx in range(0, len(p), 2)]

def calculate_lazy_theta_star(depth_prior, depth_orig, start, end, current_steps):
    map_width, map_height = depth_orig.shape[0], depth_orig.shape[1]
    # cut arm from image and set it to 0 and 1
    depth_wo_arm = np.copy(depth_orig)
    depth_wo_arm[depth_wo_arm <= 218] = 255
    depth_wo_arm[depth_wo_arm < 220] = 1
    depth_wo_arm[depth_wo_arm >= 220] = 0

    kernel_dilate = (np.ones((7,7), np.float32)) / 49
    kernel_erode = (np.ones((3, 3), np.float32)) / 9
    depth_only_obstacles = cv2.erode(depth_wo_arm.copy(), kernel_erode, iterations=2)
    depth_only_obstacles = cv2.dilate(depth_only_obstacles, kernel_dilate, iterations=3)
    depth_only_obstacles = floodfill(depth_only_obstacles.copy(), start)

    depth_only_obstacles += depth_prior
    depth_only_obstacles[depth_only_obstacles > 1] = 1
    depth_only_obstacles[depth_only_obstacles == 1] = 255
    world_map = depth_only_obstacles.flatten().tolist()
    path, length = LazyThetaStarPython.FindPath(start, end, world_map, map_width, map_height)
    path_img = get_path_img(cv2.cvtColor(depth_only_obstacles.copy(), cv2.COLOR_GRAY2RGB),  path)
    return (create_np_path(path), length,  path_img)
