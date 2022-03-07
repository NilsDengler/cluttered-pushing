#!/usr/bin/env python3
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "Lazy-Theta-with-optimization-any-angle-pathfinding/build/"))
print(os.path.join(os.path.dirname(__file__), "Lazy-Theta-with-optimization-any-angle-pathfinding/build/"))
import LazyThetaStarPython
import numpy as np
import matplotlib.pyplot as plt
import cv2
import imutils
import math
def plot_path(orig, path, x=None):
    plt_fig = plt.figure()  # figsize=(1, 2))

    plt.subplot(1, 3, 1)
    plt.imshow(path)
    plt.axis("off")
    plt.show(block=False)

    # Plot reconstruction
    plt.subplot(1, 3, 2)
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

def find_closest_center(cnts_center, sim_center):
    closest = np.inf
    return_center = None
    for c in cnts_center:
        dist = np.linalg.norm(np.asarray(sim_center)-np.asarray(c))
        if dist < closest:
            closest = dist
            return_center = c
    return return_center

def save_path_images(img, path, current_steps):
    for idx in range(0, len(path) - 2, 2):
        img = cv2.line(img, (path[idx], path[idx + 1]), (path[idx + 2], path[idx + 3]), (0, 255, 0),thickness=2)
    plt.imsave("/home/user/dengler/test_path_saving/save_" + str(current_steps) + "_orig.png", img)

def get_path_img(img, path):
    if len(path) < 4: return img
    for idx in range(0, len(path) - 2, 2):
        img = cv2.line(img, (path[idx], path[idx + 1]), (path[idx + 2], path[idx + 3]), (0, 255, 0),thickness=2)
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
    depth_only_obstacles = cv2.erode(depth_wo_arm.copy(), kernel_erode, iterations=1)
    depth_only_obstacles = cv2.dilate(depth_only_obstacles, kernel_dilate, iterations=3)
    depth_only_obstacles = floodfill(depth_only_obstacles.copy(), start)

    depth_only_obstacles += depth_prior
    depth_only_obstacles[depth_only_obstacles > 1] = 1
    depth_only_obstacles[depth_only_obstacles == 1] = 255
    world_map = depth_only_obstacles.flatten().tolist()
    path, length = LazyThetaStarPython.FindPath(start, end, world_map, map_width, map_height)
    #if not path:
    #    plt.imsave("/home/user/dengler/Documents/robot_arm_RL/pushing_net/push_gym/push_gym/theta_fails/save_" + str(current_steps) + "_orig.png", depth_orig)
    #    plt.imsave("/home/user/dengler/Documents/robot_arm_RL/pushing_net/push_gym/push_gym/theta_fails/save_" + str(current_steps) + "_filtered.png", depth_only_obstacles)
    path_img = get_path_img(cv2.cvtColor(depth_only_obstacles.copy(), cv2.COLOR_GRAY2RGB),  path)
    #rotatet_img = rotate_img_by_angle(path_img, path)
    #plt.imshow(rotatet_img)
    #plt.show()
    return (create_np_path(path), length,  path_img)

def get_angle_of_two_points(p1,p2):
    return np.arctan2(np.array([p1]), np.array([p2]))

def rotate_img_by_angle(img, path):
    if len(path) < 4: return img
    angle = get_angle_of_two_points([path[0],path[1]], [path[2],path[3]])
    #print(angle)
    rotated = imutils.rotate_bound(img, math.degrees(angle[0][0]))
    return rotated
