import matplotlib.pyplot as plt
import numpy as np
import cv2
import random
import glob
import sys, os
import math
import pybullet_data
import scipy.misc
from skimage.draw import line, polygon
from custom_utils import load_model, TURTLEBOT_URDF, joints_from_names, \
    set_joint_positions, HideOutput, get_bodies, sample_placement, pairwise_collision, \
    set_point, Point, create_box, stable_z, TAN, GREY, connect, PI, OrderedSet, \
    wait_if_gui, dump_body, set_all_color, BLUE, child_link_from_joint, link_from_name, draw_pose, Pose, pose_from_pose2d, \
    get_random_seed, get_numpy_seed, set_random_seed, set_numpy_seed, plan_joint_motion, plan_nonholonomic_motion, \
    joint_from_name, safe_zip, draw_base_limits, BodySaver, WorldSaver, LockRenderer, elapsed_time, disconnect, flatten, \
    INF, wait_for_duration, get_unbuffered_aabb, draw_aabb, DEFAULT_AABB_BUFFER, get_link_pose, get_joint_positions, \
    get_subtree_aabb, get_pairs, get_distance_fn, get_aabb, set_all_static, step_simulation, get_bodies_in_region, \
    AABB, update_scene, Profiler, pairwise_link_collision, BASE_LINK, get_collision_data, draw_pose2d, \
    normalize_interval, wrap_angle, CIRCULAR_LIMITS, wrap_interval, Euler, rescale_interval, adjust_path, WHITE, RED, \
    sample_pos_in_env, remove_body, get_euler, get_point, get_config, reset_sim, set_pose, get_quat,euler_from_quat, \
    quat_from_euler, pixel_from_point, create_cylinder, create_capsule, create_sphere
class Eval:
    def __init__(self, sim, utils):
        self.sim = sim
        self.utils = utils
        self.success_list = []
        self.reward_list = []
        self.run_length_list = []
        self.path_length_list = []
        self.current_path_length = 0
        self.current_path = []
        self.current_initial_image = None
        self.global_count = 0

    def evaluation_reset(self):
        which_obstacle = self.sim.evaluation_sample["obstacle_scenario_id"]
        self.utils.choose_obstacle_build(which_obstacle)
        self.utils.reset_obj_fix(self.sim.pushing_object_id, self.sim.evaluation_sample["start"])
        self.sim.step_simulation(self.sim.per_step_iterations)
        self.sim.current_obj_conf = get_config(self.sim.pushing_object_id, self.sim._p, self.sim.client_id)
        self.sim.last_obj_conf = self.sim.current_obj_conf
        self.sim.goal_obj_conf = self.sim.evaluation_sample['goal']
        arm_pose = self.sim.evaluation_sample['gripper']
        _, self.sim.current_depth_img, self.sim.current_true_depth = self.utils.get_image()
        self.sim.initial_true_depth = self.sim.current_true_depth
        self.sim.initial_image_processed = self.utils.process_initial_image(self.sim.current_depth_img, self.sim.current_obj_conf)
        target_joint_states = self.sim.get_ik_joints(arm_pose[0], euler_from_quat(arm_pose[1]),
                                                     self.sim._robot_tool_center)[:6]
        self.sim._reset_arm_fixed(target_joint_states)

    def write_evaluation_RL(self, Done, reward):
        if Done:
            if self.sim.target_reached:
                self.success_list.append(True)
            else: self.success_list.append(False)
            self.reward_list.append(reward)
            self.run_length_list.append(self.sim.current_steps)
            self.sim.contact_frames.append(self.sim.has_contact)
            if self.sim.save_evaluations:
                os.makedirs(self.sim.evaluation_save_path, exist_ok=True)
                self.utils.save_np(np.asarray(self.success_list), self.sim.evaluation_save_path + "success_list.npy")
                self.utils.save_np(np.asarray(self.reward_list), self.sim.evaluation_save_path + "reward_list.npy")
                self.utils.save_np(np.asarray(self.run_length_list), self.sim.evaluation_save_path + "run_length_list.npy")
                self.utils.save_np(np.asarray(self.sim.contact_frames), self.sim.evaluation_save_path + "contact_frames.npy")

    def straight_pushing_eval_RL(self, Done, current_pose, last_pose, img):
        pos_distance = np.linalg.norm(last_pose - current_pose)
        self.current_path_length += pos_distance
        self.current_path.append([current_pose, last_pose])
        if Done:
            self.path_length_list.append(self.current_path_length)
            self.straight_pushing_picture_RL(img)
            self.current_path_length = 0
            self.current_path.clear()
            self.global_count+=1

    def straight_pushing_picture_RL(self, img):
        img = self.sim.initial_path_img.copy()
        black_pixels_mask = np.all(img == [0, 0, 0], axis=-1)
        img[black_pixels_mask] = [255, 255, 255]
        for p in self.current_path:
            p_1 = self.utils.get_pos_in_image(p[0])
            p_2 = self.utils.get_pos_in_image(p[1])
            img = cv2.line(img, (p_1[0], p_1[1]), (p_2[0], p_2[1]), [255,0,0], 2)
        cv2.imwrite(self.evaluation_save_path + "path_img_" + str(self.global_count) + ".png", img)

    def write_evaluation_baseline(self, reached):
        if reached: self.success_list.append(True)
        else: self.success_list.append(False)
        if self.sim.save_evaluations:
            os.makedirs(self.sim.evaluation_save_path, exist_ok=True)
            self.utils.save_np(np.asarray(self.success_list), self.sim.evaluation_save_path + "success_list.npy")
            self.utils.save_np(np.asarray(self.current_path_length), self.sim.evaluation_save_path + "real_path_length_list.npy")
            self.utils.save_np(np.asarray(self.reward_list), self.sim.evaluation_save_path + "reward_list.npy")
            self.utils.save_np(np.asarray(self.run_length_list), self.sim.evaluation_save_path + "run_length_list.npy")
            self.utils.save_np(np.asarray(self.sim.contact_frames), self.sim.evaluation_save_path + "contact_frames.npy")


    def straight_pushing_eval_baseline(self, Done, current_pose, last_pose):
        pos_distance = np.linalg.norm(last_pose - current_pose)
        self.current_path_length += pos_distance
        self.current_path.append([current_pose, last_pose])
        if Done:
            self.path_length_list.append(self.current_path_length)
            # self.astar_deviation()
            self.straight_pushing_picture_baseline()
            self.current_path_length = 0
            self.current_path.clear()
            self.global_count+=1

    def astar_deviation(self, plot=False):
        #get trajectory in grid coords
        trajectory = []
        for points in self.current_path:
            point = self.temp((points[0] + points[1])/2)
            trajectory.append(point)
        trajectory = np.flip(np.array(trajectory, dtype=int), axis=1)

        #cut astar path when target is in reach
        goal = np.flip(self.temp(self.sim.goal_obj_conf[0]))
        distances = np.sqrt(np.square(goal[0] - self.sim.baseline.path[:, 0]) + np.square(goal[1] - self.sim.baseline.path[:, 1]))
        till = np.min(np.argwhere(distances < self.sim.baseline.target_reached_thres*256))
        path = self.sim.baseline.path[0:max(till, 1)]

        #connect edges
        edges = np.concatenate((trajectory, path))
        finish = np.transpose(np.array(line(path[-1, 0], path[-1, 1], trajectory[-1, 0], trajectory[-1, 1])))
        start = np.transpose(np.array(line(path[0, 0], path[0, 1], trajectory[0, 0], trajectory[0, 1])))
        edges = np.concatenate((edges, finish))
        edges = np.concatenate((start, edges))

        #fill space between edges
        fill = polygon(edges[:, 0], edges[:, 1])
        # num_fill = len(fill[0]) - len(np.unique(edges, axis=1))
        num_fill = len(fill[0])
        self.path_deviation = num_fill/(np.power(2., 16.)*np.power(10., -4.)) #in square centimeters
        print("number of cells: ", num_fill)
        print("space: {} cm^2".format(self.path_deviation))

        #plot
        if plot:
            base = np.zeros((256, 256))
            base[fill] = 1
            plt.imsave(self.sim.evaluation_save_path + "fill"+ str(self.global_count) + ".png", base)

    def theta_star_deviation(self, plot=False):
        path = self.sim.initial_shortest_path
        #get trajectory in grid coords
        trajectory = []
        for points in self.current_path:
            point = self.temp((points[0] + points[1])/2)
            trajectory.append(point)
        trajectory = np.flip(np.array(trajectory, dtype=int), axis=1) #???

        #connect edges
        edges = np.concatenate((trajectory, path))
        finish = np.transpose(np.array(line(path[-1, 0], path[-1, 1], trajectory[-1, 0], trajectory[-1, 1])))
        start = np.transpose(np.array(line(path[0, 0], path[0, 1], trajectory[0, 0], trajectory[0, 1])))
        edges = np.concatenate((edges, finish))
        edges = np.concatenate((start, edges))

        #fill space between edges
        fill = polygon(edges[:, 0], edges[:, 1])
        # num_fill = len(fill[0]) - len(np.unique(edges, axis=1))
        num_fill = len(fill[0])
        print("number of cells: ", num_fill)

        #plot
        if plot:
            base = np.zeros((256, 256))
            base[fill] = 1
            plt.imsave(self.sim.evaluation_save_path + "fill"+ str(self.global_count) + ".png", base)

    def temp(self,coords):
        pos = np.asarray([0, 0], dtype=int)
        # pos[1] = int((coords[0] + 0.5) * 256 / 1)  # int((coords[0] + 0.45)*255/0.9) #
        # pos[0] = int((coords[1] + 0.9) * 256 / 1)  # int((coords[1] + 0.75)*127/0.45)#
        # pos[0] = int((coords[0] + 0.5)*(self.workspace_size[0] - 1)/1) #int((coords[0] + 0.45)*255/0.9) #
        pos[0] = int((coords[0] + 0.907/2)*(256 - 1)/0.907) #int((coords[0] + 0.45)*255/0.9) #
        # pos[1] = int((coords[1] + 0.9)*(self.workspace_size[1] - 1)/1) #int((coords[1] + 0.75)*127/0.45)#
        pos[1] = int((coords[1] + 0.903)*(256 - 1)/0.903) #int((coords[1] + 0.75)*127/0.45)#
        return pos

    def rev_temp(self, coords):
        pos = np.asarray([0,0], dtype=float)
        # pos[0] = (1/2)*(1.2/256) + coords[0]/256*1 - 0.5
        # pos[1] = (1/2)*(1/256) + coords[1]/256*1 - 0.9
        # pos[0] = (1/2)*(1.2/self.workspace_size[0]) + coords[0]/self.workspace_size[0]*1 - 0.5
        pos[0] = (1/2)*(1/256) + coords[0]/256*1 - 0.907/2
        # pos[1] = (1/2)*(1/self.workspace_size[1]) + coords[1]/self.workspace_size[1]*1 - 0.9
        pos[1] = (1/2)*(1/256) + coords[1]/256*1 - 0.903
        return pos

    def straight_pushing_picture_baseline(self):
        img = self.sim.baseline.initial_path_img.copy()
        for p in self.current_path:
            p_1 = self.temp(p[0])#self.utils.get_pos_in_image(np.append(p[0], [0.025]))
            p_2 = self.temp(p[1])#self.utils.get_pos_in_image(np.append(p[1], [0.025]))
            img = cv2.line(img, (p_2[0], p_2[1]), (p_1[0], p_1[1]), [255,0,255], 2)
        cv2.imwrite(self.sim.evaluation_save_path + "path_img_"+ str(self.global_count) + ".png", img)

