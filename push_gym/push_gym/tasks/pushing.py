#!/usr/bin/python3
from ur5_environment import ArmSim
from utils import Utils
import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import gym
import math
import random
import cv2
import tensorflow as tf
from curriculum_tasks import Curriculum
from lazy_theta_star import calculate_lazy_theta_star, floodfill
from custom_utils import get_bodies, set_point, Point, Pose, get_point, get_config, reset_sim, euler_from_quat, \
    quat_from_euler
import threading
from plt_utils import plotImage
from collections import deque
from evaluation_function import Eval
from corridors import Corridors

class Pushing(ArmSim):
    def __init__(self,render=False, shared_memory=False, hz=240, use_egl=False):
        super().__init__(render=render, shared_memory=shared_memory, hz=hz, use_egl=use_egl)
        # World variables
        self.world_id = None
        self._workspace_bounds = np.array([[-0.35, 0.35], [-0.2, -0.75], [0.021, 0.021]])
        self._collision_bounds = np.array([[-0.60, 0.60], [-1., 0], [0., 1.]])
        self.utils = Utils(self, self._p)
        self.curric = Curriculum(self, self.utils, wsb=self._workspace_bounds, p_id=self._p)
        self.eval = Eval(self, self.utils)
        #get visualization of workspace
        #self.utils.show_debug_workspace(self._workspace_bounds)
        #self.utils.show_debug_workspace(self._collision_bounds)

        #define_camera
        self.image_size = 256
        self.cyaw = 180
        self.cpitch = -90
        self.croll = 0
        self.cdist = 1.1
        self.utils.define_camera_parameters(self.image_size, self.image_size, 45, 0.16, 10)
        self.utils.define_camera(cyaw=self.cyaw, cpitch=self.cpitch, croll=self.croll, cdist=self.cdist,
                                 with_rpy=True, target_pos=[0, -0.60, 0.0])
        #init obstacles
        self.obstacles = []
        #Generate pushable object
        self.pushing_object_id = -1
        self.pushing_object_id = self.utils.generate_obj()
        self.current_obj_conf = get_config(self.pushing_object_id, self._p, self.client_id)
        self.last_obj_conf = self.current_obj_conf
        #Generate first random goal
        self.goal_obj_conf = self.utils.generate_goal(self.pushing_object_id)
        self.utils.debug_gui_target(np.asarray(self.goal_obj_conf[0]))
        #self.get_relative_coodinates_in_obj_frame()
        self.step_simulation(self.per_step_iterations)
        #save initial world state
        self.world_id = self._p.saveState()
        self.set_important_variables()
        #define action and observation space for RL
        self._set_action_space()
        self._set_observation_space()
        # create baseline approach
        self.baseline = Corridors(self)

    def signal_user_input(self):
        self.thread_running = True
        i = input()
        self.Debug = not(self.Debug)
        self.thread_running = False

    def set_important_variables(self):
        self.thread_running = False
        #local_window_parameter
        self.local_window_size = 64
        self.extended_img_border_size = int(self.local_window_size/2)
        #curriculum variables
        self.curriculum_difficulty = 0
        self.change_curriculum_difficulty = False
        self.max_curriculum_iteration_steps = 0
        #RL Variables
        self.has_contact = 0
        self.n_stack = 1
        self.current_run = 0
        self.current_steps = 0
        self.global_counter = 0
        self.target_reached_thres = 0.05
        self.target_max_dist = 2
        self._max_episode_steps = 200
        #Debug Variables
        self.Debug = False
        self.single_step_training_debug = False
        self.path_debug_lines = []
        self.local_window_debug_lines = []
        self.plt_fig = None
        self.plt_obj = None
        #Evaluation Variables
        self.save_evaluations = True
        self.evaluation_sample = {}
        self.success_list = []
        self.reward_list = []
        self.run_length_list = []
        self.contact_frames = []
        #initialize queues
        self.sub_goal_queue = deque(6 * [[[0, 0, 0], [0, 0, 0, 0]]], 6)
        self.latent_space_queue = deque(4 * [list(np.zeros(32))], 4)
        self.first_object_contact = False

    def setup_for_RL(self, encoder_model, obstacle_num, arm_position, with_curriculum, evaluation_save_path=None,
                     with_evaluation=False, with_local_window=True, with_sparse_reward=False, with_fixed_obstacles=True,
                     is_eval_env=False):
        self.curriculum = with_curriculum
        self.initial_arm_position = arm_position
        self.obstacle_num = obstacle_num
        self.encoder_model = encoder_model
        self.evaluation_save_path = evaluation_save_path
        print(self.evaluation_save_path)
        self.with_evaluation = with_evaluation
        self.with_local_window = with_local_window
        self.with_sparse_reward = with_sparse_reward
        self.with_fixed_obstacles = with_fixed_obstacles
        self.is_eval_env = is_eval_env
        if self.with_local_window:
            self.reconstruction_size = self.local_window_size
        else:
            self.reconstruction_size = self.image_size

    def _set_action_space(self):
        self.arm_normalization_value = 0.05
        self.yaw_normalization_value = 0.01
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)

    def _set_observation_space(self):
        """
        Observationspace:
        latent_space: 32
        relativ position of hand: 3
        relative subgoal t-1: 2
        relative subgoal t-5: 2
        insgesammt: 39
        optional:
            bool value if  hand - object contact: 1
            bool value if  hand - object contact in cone toward theta star path: 1
            delta of object position: 2
            insgesammt: 43
        """
        pi_sqd = (math.pi * 2) + 0.01
        self.obs_bound_high = []
        self.obs_bound_low = []
        # insert latent space
        latent_bound = np.ones(32) * 10
        self.obs_bound_high.extend(list(latent_bound))
        self.obs_bound_low.extend(list(-latent_bound))
        # relative position of Hand
        hand_relative_bounds_high = np.array([5, 5, pi_sqd, pi_sqd, pi_sqd])
        self.obs_bound_high.extend(list(hand_relative_bounds_high))
        self.obs_bound_low.extend(list(-hand_relative_bounds_high))
        # insert 6D joint angle poses
        self.obs_bound_high.extend(self.ul[:6])
        self.obs_bound_low.extend(self.ll[:6])
        # relative subgoals
        hand_relative_bounds_high = np.array([5, 5])
        self.obs_bound_high.extend(list(hand_relative_bounds_high))
        self.obs_bound_low.extend(list(-hand_relative_bounds_high))
        self.obs_bound_high.extend(list(hand_relative_bounds_high))
        self.obs_bound_low.extend(list(-hand_relative_bounds_high))
        # distance object to goal
        self.obs_bound_high.extend(list([5.]))
        self.obs_bound_low.extend(list([0.]))
        # Optional
        # contact with object true or false
        self.obs_bound_high.extend(list([1]))
        self.obs_bound_low.extend(list([0]))

        assert len(self.obs_bound_high) == len(self.obs_bound_low), "high and low bound of unequal length"

        self.observ_dim = len(self.obs_bound_high)
        self.observation_space = gym.spaces.Box(np.array(self.obs_bound_high), np.array(self.obs_bound_low),
                                                dtype=np.float32)
        return

    def reset(self):
        self.current_steps = 0
        self._p.configureDebugVisualizer(self._p.COV_ENABLE_RENDERING, 0)
        self.reset_task()
        self._p.configureDebugVisualizer(self._p.COV_ENABLE_RENDERING, 1)
        return np.array(self._get_observation(), dtype=np.float32)

    def reset_task(self):
        self.is_reset = True
        self.first_object_contact = False
        # remove all obstacles from current scene
        self.utils.remove_obstacles(self.obstacles)
        self.utils.obst_diameters.clear()
        self.obstacles.clear()
        # reset sim to initial status
        reset_sim(self.world_id, self._p)
        # reset environment with evaluation sample configurations
        if self.evaluation_sample:
            self.eval.evaluation_reset()
        # reset environment random for training
        else:
            #create predefined environments

            if self.obstacle_num == 0:
                which_obstacle = random.randint(0, 9)
                if self.curriculum:
                    if self.curriculum_difficulty < 5:
                        which_obstacle = random.randint(0, 5)
                    elif self.curriculum_difficulty < 7:
                        which_obstacle = random.randint(0, 7)
                self.utils.choose_obstacle_build(which_obstacle)
            # create #obstacle_num random obstacles
            else:
                self.obstacles = self.utils.include_obstacles(self.obstacles, self.obstacle_num)

            self._reset_arm()
            # reset object init and goal position and calculate initial distances
            self.utils.reset_obj(self.pushing_object_id)
            self.step_simulation(self.per_step_iterations)
            self.current_obj_conf = get_config(self.pushing_object_id, self._p, self.client_id)
            self.last_obj_conf = self.current_obj_conf

            #sample goal in environment
            if self.curriculum:
                self.include_curriculum_learning()
            else:
                #self.goal_obj_conf = self.utils.generate_goal(self.pushing_object_id) # random
                # self.goal_obj_conf = self.utils.generate_random_goal(self.pushing_object_id, 1, 0.4) # max
                self.goal_obj_conf = self.utils.generate_random_goal(self.pushing_object_id, 2, max_distance=0.6, min_distance=0.3) #minmax
            self.utils.debug_gui_target(np.asarray(self.goal_obj_conf[0]))

            # get first image of scene
            _, self.current_depth_img, self.current_true_depth = self.utils.get_image()
            self.initial_true_depth = self.current_true_depth
            self.initial_image_processed = self.utils.process_initial_image(self.current_depth_img, self.current_obj_conf)

            # reset arm pose
            x_trans, y_trans = np.random.choice([-1, 1], 2) * np.random.uniform(0.08, 0.1, 2)
            arm_pose = np.asarray(get_point(self.pushing_object_id, self._p, self.client_id)) - [x_trans, y_trans, 0.01]
            target_joint_states = self.get_ik_joints(arm_pose, [np.pi, 0, 0], self._robot_tool_center)[:6]
            self._reset_arm(target_joint_states)

        # step simulation twice to let objects fall down
        self.step_simulation(self.per_step_iterations)
        self.step_simulation(self.per_step_iterations)

        self.tool_tip_pose = self._get_link_world_pose(self.robot_arm_id, self._robot_tool_center)
        self.initial_tool_z = self.current_obj_conf[0][2]
        self.last_tool_tip_pose = self.tool_tip_pose

        # calculate shortest path and sub goals
        self.initial_obj_to_goal_theta_dist = self._calc_theta_distances()
        self.previous_obj_to_goal_theta_dist = self.initial_obj_to_goal_theta_dist
        if self.initial_obj_to_goal_theta_dist < 0:
            return self.reset_task()
        # get distances
        self.get_relative_coodinates_in_obj_frame()
        self.initial_hand_to_obj_distance = round(
            np.linalg.norm(np.asarray(self.hand_pos_in_obj_frame[:2]) - np.asarray([0., 0.])), 5)
        self.object_sub_goal_distance_old = None
        self.hand_to_obj_distance_old = None
        self._calc_dists()
        # set collision detection to False
        self.only_obst_collision = self.utils.check_for_collision_object()
        self.ee_colission = self.utils.check_for_collision_EE()
        self.collision_detected = self.utils.oob_check(self.current_obj_conf[0][0], self.current_obj_conf[0][1])
        self.end_of_time = False
        self.has_contact = 0
        if self.current_object_goal_distance <= self.target_reached_thres or self.collision_detected:
            return self.reset_env(True)


    def _get_observation(self):
        """
        Observationspace:
        latent_space: 32
        relativ position of hand: 3
        relative subgoal t-1: 2
        relative _subgoal t-5: 2
        insgesammt: 39
        optional:
            bool value if  hand - object contact: 1
            bool value if  hand - object contact in cone toward theta star path: 1
            delta of object position: 2
            insgesammt: 43
        """
        observation = []
        # Latent space
        self.local_window, self.corners = self.utils.get_local_window(self.current_depth_img.copy(), self.current_obj_conf, self.local_window_size)
        latent_space = self.encoder_model.encoder( tf.convert_to_tensor(self.utils.process_images(self.local_window)[None, :], dtype=tf.float32))
        self.reconstruction = self.encoder_model.decoder(latent_space).numpy().reshape(self.reconstruction_size,
                                                                                       self.reconstruction_size)
        observation.extend(latent_space.numpy().tolist()[0])

        #relative current position of hand:
        observation.extend(list(np.around(np.append(self.hand_pos_in_obj_frame[:2], self.hand_orn_in_obj_frame),3)))
        #6D joints
        jointStates = self._p.getJointStates(self.robot_arm_id, self._robot_joint_indices, physicsClientId=self.client_id)
        observation.extend([round(x[0], 3) for x in jointStates])
        #relative subgoal t-1
        observation.extend(list(self.subgoal_pos_in_obj_frame1[:2]))
        # relative subgoal t-5
        observation.extend(list(self.subgoal_pos_in_obj_frame5[:2]))
        # object to goal distance
        observation.extend([self.current_object_goal_distance])
        # OPTIONAL
        # bool value if  hand - object contact
        self.contact_with_obj = self.utils.object_contact()
        if self.contact_with_obj and not self.first_object_contact:
            self.first_object_contact = True
        if self.contact_with_obj: self.has_contact += 1
        observation.extend([int(self.contact_with_obj)])

        assert len(observation) == len(self.obs_bound_low), "observation length unequal to bounds length"
        return observation

    def compute_reward(self):
        ''' Calculate reward for current step
        input:
            target_reached: True if object to goal dist is under given threshold
            collision_detected: True if collision is detected
            with_sparse_reward: True if no intermediate rewards are given
            with_shortest: True if Theta star distance schell be used as distance, False if euclidean distance shell be used
        output: reward of current step corresponding to the current situation
        '''
        r_ft = 0
        r_touch = 0
        r_lw = 0
        hand_rew = - self.utils.dist_normalized(0, self.initial_hand_to_obj_distance, self.current_hand_to_obj_distance,"hand")
        obj_rew = - self.utils.dist_normalized(0, self.initial_obj_to_goal_theta_dist, self.current_obj_to_goal_theta_dist, "obj")
        if self.target_reached: r_d = 50
        else: r_d = hand_rew + obj_rew
        if self.only_obst_collision: r_ft = -5
        if self.ee_colission: r_ft = -2
        if self.collision_detected: r_ft = - 10
        if self.contact_with_obj: r_touch = abs(hand_rew)
        if not self.utils.ee_is_in_lw(): r_lw= -2
        return r_d + r_ft + r_touch + r_lw


    def _termination(self):
        return True if self.collision_detected or self.end_of_time or self.target_reached else False

    def _apply_action(self, action):
        # create IK for cartesian action
        self._move_to_position(action)
        self.step_simulation(self.per_step_iterations)

    def step(self, action):
        if not self.thread_running:
            threading.Thread(target=self.signal_user_input).start()
        #apply cartasian action to robot and increase step
        self._apply_action(action)
        self.current_steps += 1
        #calc new tool tip pose after appling action
        self.last_tool_tip_pose = self.tool_tip_pose
        self.tool_tip_pose = self._get_link_world_pose(self.robot_arm_id, self._robot_tool_center)
        #calc current pose of object after appling action
        self.last_obj_conf = self.current_obj_conf
        self.current_obj_conf = get_config(self.pushing_object_id, self._p,  self.client_id)
        #get new camera image
        rgb_img, self.current_depth_img, self.current_true_depth = self.utils.get_image()
        #calculate theta star path and subgoals
        self.current_obj_to_goal_theta_dist = self._calc_theta_distances()
        if self.current_obj_to_goal_theta_dist < 0: self.current_obj_to_goal_theta_dist = self.previous_obj_to_goal_theta_dist
        self.previous_obj_to_goal_theta_dist = self.current_obj_to_goal_theta_dist
        #get relativ coordinates
        self.get_relative_coodinates_in_obj_frame()
        #calc current pos of all objects
        self.obstacles_pos = self.utils.get_obstacle_pos(self.obstacles, self.inv_obj_pos, self.inv_obj_orn)
        #calculate current positional and orientational distance to goal
        self._calc_dists()
        #check if target location is reache
        self.target_reached = True if self.current_object_goal_distance <= self.target_reached_thres else False
        #check for out of bounds
        oob_collision = self.utils.oob_check(self.current_obj_conf[0][0], self.current_obj_conf[0][1])
        #check for object collisions
        self.only_obst_collision = self.utils.check_for_collision_object()
        self.ee_colission = self.utils.check_for_collision_EE()
        self.collision_detected = oob_collision

        info = {'is_success': self.target_reached}
        current_observation = np.array(self._get_observation(), dtype=np.float32)
        self.end_of_time = self.current_steps > (self._max_episode_steps - 1)
        current_reward = self.compute_reward()
        termination_info = self._termination()
        if self.with_evaluation:
            self.eval.collect_evaluation_data(np.asarray(self.current_obj_conf[0]), np.asarray(self.last_obj_conf[0]),
                                              np.asarray(self.tool_tip_pose[0]), np.asarray(self.last_tool_tip_pose[0]),
                                              self.contact_with_obj, self.only_obst_collision)
            if termination_info:
                self.eval.calculate_evaluation_data_RL(self.target_reached, current_reward, self.current_steps,
                                                       np.asarray(self.current_obj_conf[0]))
                self.eval.write_evaluation(is_RL=True)
                self.global_counter += 1
        #Visualize Theta star path, depth image and reconstructed vae image
        if self.Debug:
            self.init_debug_plt(current_reward)
        self.is_reset = False
        return current_observation, current_reward, termination_info, info


    ###########################
    '''Environment functions'''
    ###########################
    def get_relative_coodinates_in_obj_frame(self):
        self.inv_obj_pos, self.inv_obj_orn = self._p.invertTransform(self.current_obj_conf[0], self.current_obj_conf[1])
        self.last_obj_conf_in_obj_frame_pos, last_obj_conf_in_obj_frame_orn = self.utils.pb_transformation(self.inv_obj_pos, self.inv_obj_orn, self.last_obj_conf)
        self.hand_pos_in_obj_frame, hand_orn_in_obj_frame = self.utils.pb_transformation(self.inv_obj_pos, self.inv_obj_orn, self.tool_tip_pose)
        self.last_hand_pos_in_obj_frame, last_hand_orn_in_obj_frame = self.utils.pb_transformation(self.inv_obj_pos,self.inv_obj_orn,self.last_tool_tip_pose)
        self.goal_pos_in_obj_frame, goal_orn_in_obj_frame = self.utils.pb_transformation(self.inv_obj_pos, self.inv_obj_orn, self.goal_obj_conf)
        self.subgoal_pos_in_obj_frame1, _ = self.utils.pb_transformation(self.inv_obj_pos, self.inv_obj_orn, self.sub_goal_queue[1])
        self.subgoal_pos_in_obj_frame5, _ = self.utils.pb_transformation(self.inv_obj_pos, self.inv_obj_orn, self.sub_goal_queue[5])

        self.goal_orn_in_obj_frame = [round(i, 2) for i in euler_from_quat(goal_orn_in_obj_frame)]
        self.hand_orn_in_obj_frame = [round(i, 2) for i in euler_from_quat(hand_orn_in_obj_frame)]
        self.last_obj_conf_in_obj_frame_orn = [round(i, 4) for i in euler_from_quat(last_obj_conf_in_obj_frame_orn)]
        self.last_hand_orn_in_obj_frame = [round(i, 4) for i in euler_from_quat(last_hand_orn_in_obj_frame)]

    def _calc_theta_distances(self):
        obj_pixel_pos = self.utils.get_pos_in_image(self.current_obj_conf[0])
        obj_pixel_pos = self.utils.pixel_boundary_check(obj_pixel_pos[0], obj_pixel_pos[1], self.image_size, self.image_size)
        goal_pixel_pos = self.utils.get_pos_in_image(self.goal_obj_conf[0])
        goal_pixel_pos = self.utils.pixel_boundary_check(goal_pixel_pos[0], goal_pixel_pos[1], self.image_size, self.image_size)
        self.current_path, obj_to_goal_theta_dist, self.path_img = calculate_lazy_theta_star(self.initial_image_processed, self.current_depth_img, list(obj_pixel_pos), list(goal_pixel_pos), self.current_steps)
        if self.current_path:
            if len(self.current_path) > 1:
                if self.is_reset:
                    self.sub_goal_queue = deque(6*[self.get_sub_goal(self.current_path)],6)
                else:
                    self.sub_goal_queue.appendleft(self.get_sub_goal(self.current_path))
            return obj_to_goal_theta_dist
        return -1, 0

    def get_theta_star_path_cone(self, path):
        start_point = self.utils.process_pixel_to_3dpoint(path[0], self.initial_true_depth , self._workspace_bounds)
        second_point = self.utils.process_pixel_to_3dpoint(path[1], self.initial_true_depth , self._workspace_bounds)
        return self.utils.check_ee_in_object_cone(second_point[:2],
                                                    start_point[:2],
                                                    self.tool_tip_pose[0][:2], 40)


    def _calc_dists(self):
        self.current_object_sub_goal_distance = round(
            np.linalg.norm(np.asarray(self.subgoal_pos_in_obj_frame1[:2]) - np.asarray([0., 0.])), 5)
        self.current_hand_to_obj_distance = round(
            np.linalg.norm(np.asarray(self.hand_pos_in_obj_frame[:2]) - np.asarray([0., 0.])), 5)
        if self.object_sub_goal_distance_old is None:
            self.object_sub_goal_distance_old = self.current_object_sub_goal_distance
        if self.hand_to_obj_distance_old is None:
            self.hand_to_obj_distance_old = self.current_hand_to_obj_distance
        self.current_object_goal_distance = round(np.linalg.norm(np.array(self.current_obj_conf[0][:2]) - np.array(self.goal_obj_conf[0][:2])),5)

    def get_sub_goal(self, path):
        path_point = self.utils.process_pixel_to_3dpoint(path[1], self.initial_true_depth, self._workspace_bounds)
        object_point = self.current_obj_conf[0]
        distance = np.linalg.norm(np.array(path_point[:2]) - np.array(object_point[:2]))
        if distance < 0.2:
            return [[path_point[0], path_point[1], self.goal_obj_conf[0][2]], list(quat_from_euler([0, 0, 0]))]
        percentage = 1 - (0.2 / distance)
        subgoal_point = self.utils.interpol_percentage_pos(np.array(path_point[:2]), object_point[:2], percentage)
        return [[subgoal_point[0], subgoal_point[1],self.goal_obj_conf[0][2]], list(quat_from_euler([0, 0, 0]))]

    def include_curriculum_learning(self):
        if self.change_curriculum_difficulty:
            if self.curriculum_difficulty < self.max_curriculum_iteration_steps - 1:
                self.curriculum_difficulty += 1
            self.change_curriculum_difficulty = False
        self.curric.close_to_far_curriculum_fixed_obstacles(self.curriculum_difficulty)

    ######################################
    '''Helper functions'''
    ######################################
    def set_point_to_bound(self, p):
        if p < 0: return 0
        elif p > self.image_size-1: return self.image_size-1
        else: return p


    def init_debug_plt(self, current_reward):
        # point local window
        sub1_center = self.utils.get_pos_in_image(self.sub_goal_queue[1][0])
        sub5_center = self.utils.get_pos_in_image(self.sub_goal_queue[5][0])
        orig_image = cv2.line(self.current_depth_img.copy(), (self.corners[0][0], self.corners[0][1]),
                              (self.corners[1][0], self.corners[1][1]), 222, 5)
        orig_image = cv2.line(orig_image, (self.corners[1][0], self.corners[1][1]),
                              (self.corners[2][0], self.corners[2][1]), 222, 5)
        orig_image = cv2.line(orig_image, (self.corners[2][0], self.corners[2][1]),
                              (self.corners[3][0], self.corners[3][1]), 222, 5)
        orig_image = cv2.line(orig_image, (self.corners[3][0], self.corners[3][1]),
                              (self.corners[0][0], self.corners[0][1]), 222, 5)
        # POINT SUBGOALS
        orig_image = cv2.circle(orig_image, tuple(sub1_center), 5, 223, -1)
        orig_image = cv2.circle(orig_image, tuple(sub5_center), 5, 224, -1)
        # show gripper action direction whole image or just direction
        arrow_x1, arrow_x2 = self.set_point_to_bound(self.last_gripper_image_points[1]), self.set_point_to_bound(
            self.current_gripper_image_points[1])
        arrow_y1, arrow_y2 = self.set_point_to_bound(self.last_gripper_image_points[0]), self.set_point_to_bound(
            self.current_gripper_image_points[0])
        norm_point_2 = 33 + (np.asarray([arrow_x2, arrow_y2]) - np.asarray([arrow_x1, arrow_y1]))
        arrowed_image_small = cv2.arrowedLine(np.ones((65, 65)) * 221, (33, 33), (norm_point_2[1], norm_point_2[0]), 222, 2)
        # plot debug text
        debug_text_image = np.ones((250, 250)) * 221
        debug_text_image = cv2.putText(debug_text_image, "Contact: " + str(int(self.contact_with_obj)), (10, 100),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.75, 223, 3, cv2.LINE_AA)
        # plot images
        self.plt_fig, self.plt_obj = plotImage(self.image_size, 5, self.plt_fig, self.plt_obj, self.path_img,
                                               orig_image, self.local_window, self.reconstruction, arrowed_image_small,
                                               debug_text_image, current_reward)  # self.reconstruction
