#!/usr/bin/python3
from scipy.spatial.transform import Rotation as R
from arm_sim import ArmSim
from pybullet_cam import PyCam as Cam
from pybullet_environment_building import EnvBuild as build
from pybullet_collision_checker import ColCheck as collision
from debug_points import SphereMarker
import matplotlib
matplotlib.use('TkAgg')
#import warnings
#warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import numpy as np
import os
import random
from tqdm import tqdm
import cv2
import tensorflow as tf
from custom_utils import load_model, TURTLEBOT_URDF, joints_from_names, \
    set_joint_positions, HideOutput, get_bodies, sample_placement, pairwise_collision, \
    set_point, Point, create_box, stable_z, TAN, GREY, connect, PI, OrderedSet, \
    wait_if_gui, dump_body, set_all_color, BLUE, child_link_from_joint, link_from_name, draw_pose, Pose, pose_from_pose2d, \
    get_random_seed, get_numpy_seed, set_random_seed, set_numpy_seed, plan_joint_motion, plan_nonholonomic_motion, \
    joint_from_name, safe_zip, draw_base_limits, BodySaver, WorldSaver, LockRenderer, elapsed_time, disconnect, flatten, \
    INF, wait_for_duration, get_unbuffered_aabb, draw_aabb, DEFAULT_AABB_BUFFER, get_link_pose, get_joint_positions, \
    get_subtree_aabb, get_pairs, get_distance_fn, get_aabb, set_all_static, step_simulation, get_bodies_in_region, \
    AABB, update_scene, Profiler, pairwise_link_collision, BASE_LINK, get_collision_data, draw_pose2d, \
    normalize_interval, wrap_angle, CIRCULAR_LIMITS, wrap_interval, Euler, rescale_interval, adjust_path, WHITE, \
    sample_pos_in_env, remove_body, get_euler, get_point, get_config, reset_sim, set_pose, get_quat,euler_from_quat, \
    quat_from_euler, pixel_from_point, set_client, RED, create_cylinder
import threading
from plt_utils import plotImage
from collections import deque

import json
import yaml
from yaml.loader import SafeLoader

UR5_URDF_PATH = "../../robots/assets/ur5_with_gripper/ur5.urdf"
UR5_WORKSPACE_URDF_PATH = '../../robots/assets/ur5_with_gripper/workspace.urdf'
PLANE_URDF_PATH = '../../robots/assets/plane/sand_plane.urdf'

class ArmEnvSim(ArmSim):
    def __init__(self, render=False):
        super().__init__(render=render)
        #set observation and action space
        # World variables
        self.world_id = None
        self._workspace_bounds = np.array([[-0.35, 0.35],  # 3x2 rows: x,y,z cols: min,max
                                        [-0.3, -0.75],[0.02, 0.02]])
        self._collision_bounds = np.array([[-0.60, 0.60],  # 3x2 rows: x,y,z cols: min,max
                                        [-1., 0],[0., 1.]])
        self.current_run = 0
        self.current_steps = 0
        self.target_reached_thres = 0.05
        self.target_max_dist = 2
        self.reward_value = 1.0
        self._max_episode_steps = 150

        # Camera parameters
        self.image_size = 256
        self.fov = 45
        self.near = 0.16
        self.far = 10
        self.cyaw = 180
        self.cpitch = -90
        self.croll =  0
        self.cdist = 1.1

        #local_window_parameter
        self.local_window_size = 64
        self.extended_img_border_size = int(self.local_window_size/2)

        self._target_dist_min = 0.05
        self.x_id, self.y_id, self.z_id = -1, -1, -1
        self.with_orientation = False
        self.with_latent = False
        self.num_reached_goals = 0
        self.current_pixel_coordinates = []
        self.last_pixel_coordinates = []

        self.testing_msg = 0
        self.plt_fig = None
        self.save_plt_fig = None
        self.plt_obj = None
        self.curriculum_difficulty = 0
        self.change_curriculum_difficulty = False
        self.max_curriculum_iteration_steps = 0

        self.Debug = False
        self.single_step_training_debug = False
        self.thread_running = False
        self.path_debug_lines = []
        self.local_window_debug_lines = []
        self.success_list = []
        self.reward_list = []
        self.run_length_list =  []
        self.save_evaluations = True

        self.obj_pos_in_target = None
        self.sub_goal_queue = deque(6*[[[0,0,0],[0,0,0,0]]],6)
        self.latent_space_queue = deque(4*[list(np.zeros(32))],4)
        self.n_stack = 1
        self.obj_config = "random_obj"
        self.goal_config = "random_goal"
        self.target_reached_thres = 0.06

        self.object_diameter = 0.06
        self.arm_diameter = self.object_diameter#*3/4
        self._p.resetDebugVisualizerCamera(
            cameraDistance=self.cdist,
            cameraYaw=self.cyaw,
            cameraPitch=self.cpitch,
            cameraTargetPosition=[0, -0.6, 0],
        )
        self.generate_workspace()

    def signal_user_input(self):
        self.thread_running = True
        i = input()  # I have python 2.7, not 3.x
        self.Debug = not(self.Debug)
        self.thread_running = False

    ###########################
    '''Environment functions'''
    ###########################

    def generate_workspace(self):
        self.pushing_object_id = -1
        self.cam = Cam(self.image_size, self.image_size, self.fov, self.near, self.far, self, self._p)
        self.col = collision(self, self._p)
        self.builder = build(self, self.col, self.obj_config, self.goal_config, wsb =self._workspace_bounds, p_id = self._p)
        self.obstacles = []
        # create table plate
        self._p.loadURDF(os.path.join(os.path.dirname(__file__), PLANE_URDF_PATH), [0, 0, -0.001])
        self.table = self._p.loadURDF(os.path.join(os.path.dirname(__file__), UR5_WORKSPACE_URDF_PATH))
        self._p.changeVisualShape(self.table, -1, rgbaColor=[210/255, 183/255, 115/255, 1], physicsClientId=self.client_id)
        # load robot arm and gripper
        self._load_robot_complete()
        #self._load_robot(with_arm)
        self.initial_pose = self.get_ik_joints([0.0, -0.198, 0.021], [np.pi, 0, 0], self._robot_tool_center)[:6]
        self._reset_arm()
        self.tool_tip_pose = self._get_link_world_pose(self.robot_arm_id, self._robot_tool_center)
        self.last_tool_tip_pose = self.tool_tip_pose
        #generate pushing object
        # pushing fragment
        BLOCK_URDF_PATH = '../../robots/assets/fragments/frag1.urdf'
        xpos, ypos, zpos = [0.05, -0.3, 0.031]
        yaw = self._p.getQuaternionFromEuler([0, 0, 0])
        self.pushing_object_id = self._p.loadURDF(os.path.join(os.path.dirname(__file__), BLOCK_URDF_PATH),
                                    basePosition=[xpos, ypos, zpos], baseOrientation=list(yaw),
                                    physicsClientId=self.client_id, globalScaling=1.5, useFixedBase=True)
        self._p.changeVisualShape(self.pushing_object_id, -1, rgbaColor=[156 / 255, 107 / 255, 48 / 255, 1],
                                  physicsClientId=self.client_id)
        #self.pushing_object_id = self.builder.generate_obj()
        self.init_obj_conf = get_config(self.pushing_object_id, self._p, self.client_id)
        self.current_obj_conf = self.init_obj_conf
        self.last_obj_conf = self.current_obj_conf

        #include other urdfs
        #long block
        #BLOCK_URDF_PATH = '../../robots/assets/block/block_long.urdf'
        #xpos, ypos, zpos = [-0.15, -0.5, 0.001]
        #yaw = self._p.getQuaternionFromEuler([0, 0, np.pi/2])
        #long_block_id = self._p.loadURDF(os.path.join(os.path.dirname(__file__), BLOCK_URDF_PATH),
        #                          basePosition=[xpos, ypos, zpos], baseOrientation=list(yaw),
        #                          physicsClientId=self.client_id)
        #block
        #BLOCK_URDF_PATH = '../../robots/assets/block/block.urdf'
        #xpos, ypos, zpos = [0, -0.5, 0.001]
        #yaw = self._p.getQuaternionFromEuler([0, 0, 0])
        #block_id = self._p.loadURDF(os.path.join(os.path.dirname(__file__), BLOCK_URDF_PATH),
        #                          basePosition=[xpos, ypos, zpos], baseOrientation=list(yaw),
        #                          physicsClientId=self.client_id)
        #cylinder
        #cylinder_id = create_cylinder(self._p, self.client_id, 0.03, 0.06, mass=.1, color=BLUE, pose=[[0.15, -0.5, 0.001], self._p.getQuaternionFromEuler([0, 0, 0])])
        self.step_simulation(self.per_step_iterations)
        self.world_id = self._p.saveState()

        # borders
        BLOCK_URDF_PATH = '../../robots/assets/objects/border.urdf'
        xpos, ypos, zpos = [0.45, -0.5, 0.031]
        yaw = self._p.getQuaternionFromEuler([0,0, np.pi/2])
        border1 = self._p.loadURDF(os.path.join(os.path.dirname(__file__), BLOCK_URDF_PATH),
                                    basePosition=[xpos, ypos, zpos], baseOrientation=list(yaw),
                                    physicsClientId=self.client_id,globalScaling=1.5, useFixedBase=True)
        self._p.changeVisualShape(border1, -1, rgbaColor=[127/255,53/255,11/255, 1], physicsClientId=self.client_id)

        BLOCK_URDF_PATH = '../../robots/assets/objects/border.urdf'
        xpos, ypos, zpos = [-0.45, -0.5, 0.031]
        yaw = self._p.getQuaternionFromEuler([0,0, np.pi/2])
        border2 = self._p.loadURDF(os.path.join(os.path.dirname(__file__), BLOCK_URDF_PATH),
                                    basePosition=[xpos, ypos, zpos], baseOrientation=list(yaw),
                                    physicsClientId=self.client_id,globalScaling=1.5, useFixedBase=True)
        self._p.changeVisualShape(border2, -1, rgbaColor=[127/255,53/255,11/255, 1], physicsClientId=self.client_id)

        BLOCK_URDF_PATH = '../../robots/assets/objects/border.urdf'
        xpos, ypos, zpos = [-0.0, -0.95, 0.031]
        yaw = self._p.getQuaternionFromEuler([0, 0, 0])
        border3 = self._p.loadURDF(os.path.join(os.path.dirname(__file__), BLOCK_URDF_PATH),
                                   basePosition=[xpos, ypos, zpos], baseOrientation=list(yaw),
                                   physicsClientId=self.client_id, globalScaling=1.5, useFixedBase=True)
        self._p.changeVisualShape(border3, -1, rgbaColor=[127/255,53/255,11/255, 1], physicsClientId=self.client_id)
        # #fragment

        #goal fragment
        BLOCK_URDF_PATH = '../../robots/assets/fragments/frag1.urdf'
        xpos, ypos, zpos = [-0.1875, -0.64, 0.031]
        yaw = self._p.getQuaternionFromEuler([0,0, np.deg2rad(-30)])
        self.goal_fragment = self._p.loadURDF(os.path.join(os.path.dirname(__file__), BLOCK_URDF_PATH),
                                    basePosition=[xpos, ypos, zpos], baseOrientation=list(yaw),
                                    physicsClientId=self.client_id,globalScaling=1.5, useFixedBase=True)
        self._p.changeVisualShape( self.goal_fragment, -1, rgbaColor=[0/255, 255/255, 0/255, 0.75], physicsClientId=self.client_id)
        #shell
        BLOCK_URDF_PATH = '../../robots/assets/fragments/frag2.urdf'
        xpos, ypos, zpos = [0, -0.71, 0.031]
        yaw = self._p.getQuaternionFromEuler([0, 0, 0])
        block_id2 = self._p.loadURDF(os.path.join(os.path.dirname(__file__), BLOCK_URDF_PATH),
                                    basePosition=[xpos, ypos, zpos], baseOrientation=list(yaw),
                                    physicsClientId=self.client_id,globalScaling=1.5, useFixedBase=True)
        self._p.changeVisualShape(block_id2, -1, rgbaColor=[156/255, 107/255, 48/255, 1], physicsClientId=self.client_id)

        BLOCK_URDF_PATH = '../../robots/assets/fragments/frag3.urdf'
        xpos, ypos, zpos = [0, -0.71, 0.031]
        yaw = self._p.getQuaternionFromEuler([0,0, 0])
        block_id3 = self._p.loadURDF(os.path.join(os.path.dirname(__file__), BLOCK_URDF_PATH),
                                    basePosition=[xpos, ypos, zpos], baseOrientation=list(yaw),
                                    physicsClientId=self.client_id,globalScaling=1.5, useFixedBase=True)
        self._p.changeVisualShape(block_id3, -1, rgbaColor=[156/255, 107/255, 48/255, 1], physicsClientId=self.client_id)

        BLOCK_URDF_PATH = '../../robots/assets/fragments/frag5.urdf'
        xpos, ypos, zpos = [0, -0.71, 0.031]
        yaw = self._p.getQuaternionFromEuler([0,0, 0])
        block_id5 = self._p.loadURDF(os.path.join(os.path.dirname(__file__), BLOCK_URDF_PATH),
                                    basePosition=[xpos, ypos, zpos], baseOrientation=list(yaw),
                                    physicsClientId=self.client_id,globalScaling=1.5)
        self._p.changeVisualShape(block_id5, -1, rgbaColor=[156/255, 107/255, 48/255, 1], physicsClientId=self.client_id)

        BLOCK_URDF_PATH = '../../robots/assets/fragments/frag6.urdf'
        xpos, ypos, zpos = [-0.1, -0.71, 0.031]
        yaw = self._p.getQuaternionFromEuler([0,0, 0])
        block_id6 = self._p.loadURDF(os.path.join(os.path.dirname(__file__), BLOCK_URDF_PATH),
                                    basePosition=[xpos, ypos, zpos], baseOrientation=list(yaw),
                                    physicsClientId=self.client_id,globalScaling=1.5)
        self._p.changeVisualShape(block_id6, -1, rgbaColor=[156/255, 107/255, 48/255, 1], physicsClientId=self.client_id)

        BLOCK_URDF_PATH = '../../robots/assets/fragments/frag7.urdf'
        xpos, ypos, zpos = [-0.2, -0.71, 0.031]
        # yaw = self._p.getQuaternionFromEuler([np.pi/4, + np.pi, 0])
        yaw = self._p.getQuaternionFromEuler([0, 0, 0])
        block_id7 = self._p.loadURDF(os.path.join(os.path.dirname(__file__), BLOCK_URDF_PATH),
                                     basePosition=[xpos, ypos, zpos], baseOrientation=list(yaw),
                                     physicsClientId=self.client_id, globalScaling=1.5)
        self._p.changeVisualShape(block_id7, -1, rgbaColor=[156 / 255, 107 / 255, 48 / 255, 1],
                                  physicsClientId=self.client_id)

        #extra clutter
        BLOCK_URDF_PATH = '../../robots/assets/fragments/frag6.urdf'
        xpos, ypos, zpos = [-0.3, -0.3, 0.031]
        yaw = self._p.getQuaternionFromEuler([0, 0, np.pi/3.5])
        block_id6 = self._p.loadURDF(os.path.join(os.path.dirname(__file__), BLOCK_URDF_PATH),
                                     basePosition=[xpos, ypos, zpos], baseOrientation=list(yaw),
                                     physicsClientId=self.client_id, globalScaling=1.5)
        self._p.changeVisualShape(block_id6, -1, rgbaColor=[156 / 255, 107 / 255, 48 / 255, 1],
                                  physicsClientId=self.client_id)

        BLOCK_URDF_PATH = '../../robots/assets/fragments/frag6.urdf'
        xpos, ypos, zpos = [-0.35, -0.4, 0.031]
        yaw = self._p.getQuaternionFromEuler([0, 0, 0])
        block_id6 = self._p.loadURDF(os.path.join(os.path.dirname(__file__), BLOCK_URDF_PATH),
                                     basePosition=[xpos, ypos, zpos], baseOrientation=list(yaw),
                                     physicsClientId=self.client_id, globalScaling=1.5)
        self._p.changeVisualShape(block_id6, -1, rgbaColor=[156 / 255, 107 / 255, 48 / 255, 1],
                                  physicsClientId=self.client_id)

        BLOCK_URDF_PATH = '../../robots/assets/fragments/frag6.urdf'
        xpos, ypos, zpos = [-0.35, -0.475, 0.031]
        yaw = self._p.getQuaternionFromEuler([0, 0, 7*np.pi/5])
        block_id6 = self._p.loadURDF(os.path.join(os.path.dirname(__file__), BLOCK_URDF_PATH),
                                     basePosition=[xpos, ypos, zpos], baseOrientation=list(yaw),
                                     physicsClientId=self.client_id, globalScaling=1.5)
        self._p.changeVisualShape(block_id6, -1, rgbaColor=[156 / 255, 107 / 255, 48 / 255, 1],
                                  physicsClientId=self.client_id)

        BLOCK_URDF_PATH = '../../robots/assets/fragments/frag5.urdf'
        xpos, ypos, zpos = [-0.275, -0.675, 0.031]
        yaw = self._p.getQuaternionFromEuler([0, 0, 7*np.pi / 6 + np.deg2rad(2)])
        block_id5 = self._p.loadURDF(os.path.join(os.path.dirname(__file__), BLOCK_URDF_PATH),
                                     basePosition=[xpos, ypos, zpos], baseOrientation=list(yaw),
                                     physicsClientId=self.client_id, globalScaling=1.5)
        self._p.changeVisualShape(block_id5, -1, rgbaColor=[156 / 255, 107 / 255, 48 / 255, 1],
                                  physicsClientId=self.client_id)

        BLOCK_URDF_PATH = '../../robots/assets/fragments/frag5.urdf'
        xpos, ypos, zpos = [-0.325, -0.65, 0.031]
        yaw = self._p.getQuaternionFromEuler([0, 0, np.deg2rad(20) - np.deg2rad(20)])
        block_id5 = self._p.loadURDF(os.path.join(os.path.dirname(__file__), BLOCK_URDF_PATH),
                                     basePosition=[xpos, ypos, zpos], baseOrientation=list(yaw),
                                     physicsClientId=self.client_id, globalScaling=1.5)
        self._p.changeVisualShape(block_id5, -1, rgbaColor=[156 / 255, 107 / 255, 48 / 255, 1],
                                  physicsClientId=self.client_id)

        BLOCK_URDF_PATH = '../../robots/assets/fragments/frag7.urdf'
        xpos, ypos, zpos = [-0.3, -0.75, 0.031]
        # yaw = self._p.getQuaternionFromEuler([np.pi/4, + np.pi, 0])
        yaw = self._p.getQuaternionFromEuler([0, 0, 0])
        block_id7 = self._p.loadURDF(os.path.join(os.path.dirname(__file__), BLOCK_URDF_PATH),
                                     basePosition=[xpos, ypos, zpos], baseOrientation=list(yaw),
                                     physicsClientId=self.client_id, globalScaling=1.5)
        self._p.changeVisualShape(block_id7, -1, rgbaColor=[156 / 255, 107 / 255, 48 / 255, 1],
                                  physicsClientId=self.client_id)

        BLOCK_URDF_PATH = '../../robots/assets/fragments/frag5.urdf'
        xpos, ypos, zpos = [0.25, -0.45, 0.031]
        yaw = self._p.getQuaternionFromEuler([0,0, np.pi/6])
        block_id5 = self._p.loadURDF(os.path.join(os.path.dirname(__file__), BLOCK_URDF_PATH),
                                    basePosition=[xpos, ypos, zpos], baseOrientation=list(yaw),
                                    physicsClientId=self.client_id,globalScaling=1.5)
        self._p.changeVisualShape(block_id5, -1, rgbaColor=[156/255, 107/255, 48/255, 1], physicsClientId=self.client_id)

        #BLOCK_URDF_PATH = '../../robots/assets/fragments/frag4.urdf'
        #xpos, ypos, zpos = [0.25, -0.525, 0.031]
        #yaw = self._p.getQuaternionFromEuler([0,0, 0])
        # block_id4 = self._p.loadURDF(os.path.join(os.path.dirname(__file__), BLOCK_URDF_PATH),
        #                            basePosition=[xpos, ypos, zpos], baseOrientation=list(yaw),
        #                            physicsClientId=self.client_id,globalScaling=1.5)
        #self._p.changeVisualShape(block_id4, -1, rgbaColor=[156/255, 107/255, 48/255, 1], physicsClientId=self.client_id)

        BLOCK_URDF_PATH = '../../robots/assets/fragments/frag7.urdf'
        xpos, ypos, zpos = [0.25, -0.675, 0.031]
        # yaw = self._p.getQuaternionFromEuler([np.pi/4, + np.pi, 0])
        yaw = self._p.getQuaternionFromEuler([0, 0, 3*np.pi/4])
        block_id7 = self._p.loadURDF(os.path.join(os.path.dirname(__file__), BLOCK_URDF_PATH),
                                     basePosition=[xpos, ypos, zpos], baseOrientation=list(yaw),
                                     physicsClientId=self.client_id, globalScaling=1.5)
        self._p.changeVisualShape(block_id7, -1, rgbaColor=[156 / 255, 107 / 255, 48 / 255, 1],
                                  physicsClientId=self.client_id)

        BLOCK_URDF_PATH = '../../robots/assets/fragments/frag7.urdf'
        xpos, ypos, zpos = [0.05, -0.85, 0.031]
        # yaw = self._p.getQuaternionFromEuler([np.pi/4, + np.pi, 0])
        yaw = self._p.getQuaternionFromEuler([0, 0, 5.5*np.pi / 4])
        block_id7 = self._p.loadURDF(os.path.join(os.path.dirname(__file__), BLOCK_URDF_PATH),
                                     basePosition=[xpos, ypos, zpos], baseOrientation=list(yaw),
                                     physicsClientId=self.client_id, globalScaling=1.5)
        self._p.changeVisualShape(block_id7, -1, rgbaColor=[156 / 255, 107 / 255, 48 / 255, 1],
                                  physicsClientId=self.client_id)

        BLOCK_URDF_PATH = '../../robots/assets/fragments/frag7.urdf'
        xpos, ypos, zpos = [-0.12, -0.49, 0.031]
        # yaw = self._p.getQuaternionFromEuler([np.pi/4, + np.pi, 0])
        yaw = self._p.getQuaternionFromEuler([0, 0, 0])
        block_id7 = self._p.loadURDF(os.path.join(os.path.dirname(__file__), BLOCK_URDF_PATH),
                                     basePosition=[xpos, ypos, zpos], baseOrientation=list(yaw),
                                     physicsClientId=self.client_id, globalScaling=1.5)
        self._p.changeVisualShape(block_id7, -1, rgbaColor=[156 / 255, 107 / 255, 48 / 255, 1],
                                  physicsClientId=self.client_id)

        BLOCK_URDF_PATH = '../../robots/assets/fragments/frag4.urdf'
        xpos, ypos, zpos = [-0.07, -0.48, 0.031]
        yaw = self._p.getQuaternionFromEuler([0,0, 3*np.pi/4])
        block_id4 = self._p.loadURDF(os.path.join(os.path.dirname(__file__), BLOCK_URDF_PATH),
                                    basePosition=[xpos, ypos, zpos], baseOrientation=list(yaw),
                                    physicsClientId=self.client_id,globalScaling=1.25)
        self._p.changeVisualShape(block_id4, -1, rgbaColor=[156/255, 107/255, 48/255, 1], physicsClientId=self.client_id)


    ######################################
    '''Reinforcement learning functions'''
    ######################################

    def sample_data(self, sample_num, demo=False):
        data = {}
        data["samples"] = []
        for _ in tqdm(range(sample_num)):
            start, goal, gripper, obstacle_scenario_id = self.sample_start_goal_gripper_pos(demo=demo)
            data["samples"].append({
                "start": start,
                "goal": goal,
                "gripper": gripper,
                "obstacle_scenario_id": obstacle_scenario_id
            })
        with open('test.txt', 'w') as outfile:
            json.dump(data, outfile)

    def sample_start_goal_gripper_pos(self, demo=False):
        self.builder.remove_obstacles()
        self.obstacles.clear()
        # reset sim to initial status
        reset_sim(self.world_id, self._p)
        self._reset_arm()
        #choose Obstacle scenario
        #obstacle_scenario_id = random.randint(0, 9)
        obstacle_scenario_id = 10
        self.builder.choose_obstacle_build(obstacle_scenario_id)
        #sample new starting position
        self.builder.reset_obj(self.pushing_object_id)
        self.step_simulation(self.per_step_iterations)
        start_position = get_config(self.pushing_object_id, self._p, self.client_id)
        # sample goal position
        if demo:
            goal_position = get_config(self.goal_fragment, self._p, self.client_id)
        else:
            goal_position = self.builder.generate_random_goal(self.pushing_object_id, 2)
        # sample gripper position
        #x_trans, y_trans = random.uniform(0.08, 0.1) * random.choice([-1, 1]), \
        #                   random.uniform(0.08,0.1) * random.choice([-1, 1])
        #arm_pose = np.asarray(get_point(self.pushing_object_id, self._p, self.client_id)) - [x_trans, y_trans, 0.01]
        arm_pose = self.get_arm_to_position(start_position[0], goal_position[0])
        target_joint_states = self.get_ik_joints(arm_pose, [np.pi, 0, 0], self._robot_tool_center)[:6]
        self._reset_arm_fixed(target_joint_states)
        self.step_simulation(self.per_step_iterations)
        self.step_simulation(self.per_step_iterations)
        gripper_starting_position = self._get_link_world_pose(self.robot_arm_id, self._robot_tool_center)

        return start_position, goal_position, gripper_starting_position, obstacle_scenario_id


    def get_arm_to_position(self, start, goal):
        line = np.array(start) - np.array(goal)
        straight_start = start + line/np.linalg.norm(line)*self.arm_diameter*1.5
        goal_pose = [straight_start[0], straight_start[1], 0.021]
        return goal_pose


    def reset(self):
        pass

    def step(self, action):
        pass


    ######################################
    '''Helper functions'''
    ######################################
    def set_point_to_bound(self, p):
        if p < 0: return 0
        elif p > self.image_size-1: return self.image_size-1
        else: return p


    def evaluation(self, Done, reward):
        if Done:
            if self.target_reached:
                self.success_list.append(True)
            else: self.success_list.append(False)
            self.reward_list.append(reward)
            self.run_length_list.append(self.current_steps)
            if self.save_evaluations:
                os.makedirs(self.evaluation_save_path, exist_ok=True)
                self.cam.save_np(np.asarray(self.success_list), self.evaluation_save_path + "success_list.npy")
                self.cam.save_np(np.asarray(self.reward_list), self.evaluation_save_path + "reward_list.npy")
                self.cam.save_np(np.asarray(self.run_length_list), self.evaluation_save_path + "run_length_list.npy")

    ######################################
    '''Debug functions'''
    ######################################
    def init_debug_plt(self, current_reward):
        # point local window
        pixel_points = self.cam.get_pos_in_image(self.current_obj_conf[0])
        x1, x2 = self.set_point_to_bound(pixel_points[0] - 32), self.set_point_to_bound(pixel_points[0] + 32)
        y1, y2 = self.set_point_to_bound(pixel_points[1] - 32), self.set_point_to_bound(pixel_points[1] + 32)
        sub1_center = self.cam.get_pos_in_image(self.sub_goal_queue[1][0])
        sub5_center = self.cam.get_pos_in_image(self.sub_goal_queue[5][0])
        orig_image = cv2.rectangle(self.current_depth_img.copy(), (x1, y1), (x2, y2), 222, 3)
        # POINT SUBGOALS
        orig_image = cv2.circle(orig_image, tuple(sub1_center), 5, 223, -1)
        orig_image = cv2.circle(orig_image, tuple(sub5_center), 5, 224, -1)
        # show gripper action direction wwhole image or just direction
        arrow_x1, arrow_x2 = self.set_point_to_bound(self.old_gripper_image_points[1]), self.set_point_to_bound(
            self.new_gripper_image_points[1])
        arrow_y1, arrow_y2 = self.set_point_to_bound(self.old_gripper_image_points[0]), self.set_point_to_bound(
            self.new_gripper_image_points[0])
        norm_point_2 = 33 + (np.asarray([arrow_x2, arrow_y2]) - np.asarray([arrow_x1, arrow_y1]))
        arrowed_image = cv2.arrowedLine(np.ones((self.image_size, self.image_size)) * 221, (arrow_x1, arrow_y1),
                                        (arrow_x2, arrow_y2), 222, 5)
        arrowed_image_small = cv2.arrowedLine(np.ones((65, 65)) * 221, (33, 33), (norm_point_2[1], norm_point_2[0]),
                                              222, 2)
        # plot debug text
        debug_text_image = np.ones((250, 250)) * 221
        debug_text_image = cv2.putText(debug_text_image, "Contact: " + str(int(self.contact_with_obj)), (10, 100),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.75, 223, 3, cv2.LINE_AA)
        debug_text_image = cv2.putText(debug_text_image, "Contact in Cone: " + str(int(self.contact_with_obj_in_cone)),
                                       (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.75, 223, 3, cv2.LINE_AA)
        self.obj_distances = [-1]
        debug_text_image = cv2.putText(debug_text_image, "closest dis: " + str(round(min(self.obj_distances), 4)),
                                       (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.65, 223, 3, cv2.LINE_AA)
        # plot images
        self.plt_fig, self.plt_obj = plotImage(self.image_size, 5, self.plt_fig, self.plt_obj, self.path_img,
                                               orig_image, self.local_window, np.zeros((64,64)), arrowed_image_small,
                                               debug_text_image, current_reward)  # self.reconstruction

    def debug_local_window(self):
        self.remove_debug_lines()
        self.local_window_debug_lines.clear()
        obj_pixel_pos = self.cam.get_pos_in_image(self.current_obj_conf[0])
        obj_pixel_pos[0] -= 32
        obj_pixel_pos[1] -= 32
        lu = self.cam.process_pixel_to_3dpoint(obj_pixel_pos, self.initial_true_depth, self._workspace_bounds)
        obj_pixel_pos[1] += 64
        lo = self.cam.process_pixel_to_3dpoint(obj_pixel_pos, self.initial_true_depth, self._workspace_bounds)
        obj_pixel_pos[0] += 64
        ro = self.cam.process_pixel_to_3dpoint(obj_pixel_pos, self.initial_true_depth, self._workspace_bounds)
        obj_pixel_pos[1] -= 64
        ru = self.cam.process_pixel_to_3dpoint(obj_pixel_pos, self.initial_true_depth, self._workspace_bounds)
        self.local_window_debug_lines.append(self._p.addUserDebugLine(lu, lo, [0, 0, 1], 1.0))
        self.local_window_debug_lines.append(self._p.addUserDebugLine(lo, ro, [0, 0, 1], 1.0))
        self.local_window_debug_lines.append(self._p.addUserDebugLine(ro, ru, [0, 0, 1], 1.0))
        self.local_window_debug_lines.append(self._p.addUserDebugLine(ru, lu, [0, 0, 1], 1.0))

    def _debug_vis_path(self, path):
        self.remove_debug_lines()
        self.path_debug_lines.clear()
        for idx in range(0, len(path) - 1):
            world_points1 = self.cam.process_pixel_to_3dpoint(path[idx], self.initial_true_depth, self._workspace_bounds)
            world_points2 = self.cam.process_pixel_to_3dpoint(path[idx + 1], self.initial_true_depth , self._workspace_bounds)
            self.path_debug_lines.append(self._p.addUserDebugLine(world_points1, world_points2, [0, 1, 0], 1.0))

    def remove_debug_lines(self):
        for i in self.path_debug_lines:
            self._p.removeUserDebugItem(i)
        for i in self.local_window_debug_lines:
            self._p.removeUserDebugItem(i)
        return

    def debug_gui_target(self, t_pose, ori):
        #r = R.from_euler('xyz',ori, degrees=False)
        #t_pose = np.matmul(pose, r.as_matrix())
        if self.x_id < 0:
            self.x_id = self._p.addUserDebugLine(t_pose, [t_pose[0] + 0.1, t_pose[1], t_pose[2]], [1, 0, 0],physicsClientId=self.client_id)
            self.y_id = self._p.addUserDebugLine(t_pose, [t_pose[0], t_pose[1] + 0.1, t_pose[2]], [0, 1, 0],physicsClientId=self.client_id)
            self.z_id = self._p.addUserDebugLine(t_pose, [t_pose[0], t_pose[1], t_pose[2] + 0.1], [0, 0, 1],physicsClientId=self.client_id)
        else:
            self._p.addUserDebugLine(t_pose, [t_pose[0] + 0.1, t_pose[1], t_pose[2]], [1, 0, 0],replaceItemUniqueId=self.x_id,
                               physicsClientId=self.client_id)
            self._p.addUserDebugLine(t_pose, [t_pose[0], t_pose[1] + 0.1, t_pose[2]], [0, 1, 0],replaceItemUniqueId=self.y_id,
                               physicsClientId=self.client_id)
            self._p.addUserDebugLine(t_pose, [t_pose[0], t_pose[1], t_pose[2] + 0.1], [0, 0, 1],replaceItemUniqueId=self.z_id,
                               physicsClientId=self.client_id)


    def debug_gui_object(self, obj):
        self._p.addUserDebugLine([0, 0, 0], [0.1, 0, 0], [1, 0, 0], parentObjectUniqueId=obj, physicsClientId=self.client_id)
        self._p.addUserDebugLine([0, 0, 0], [0, 0.1, 0], [0, 1, 0], parentObjectUniqueId=obj, physicsClientId=self.client_id)
        self._p.addUserDebugLine([0, 0, 0], [0, 0, 0.1], [0, 0, 1], parentObjectUniqueId=obj, physicsClientId=self.client_id)

    ######################################
    '''legacy methods'''
    ######################################
    def calculate_astar_reward(self):
        latent_space = self.astar_model.encoder(tf.convert_to_tensor(
            self.cam.process_images(self.old_depth_img)[None, :], dtype=tf.float32))
        self.last_pixel_coordinates = self.current_pixel_coordinates.copy()
        self.current_pixel_coordinates.clear()
        self.current_pixel_coordinates.extend(self.cam.get_pos_in_image(self.current_obj_conf[0]))
        self.current_pixel_coordinates.extend(self.cam.get_pos_in_image(self.goal_obj_conf[0]))
        extended_latent = tf.concat((latent_space, tf.convert_to_tensor(
            np.asarray(self.current_pixel_coordinates).reshape((1, 4)) / 255, dtype=tf.float32)), axis=1)
        self.astar_map = self.astar_model.decoder(extended_latent).numpy().reshape(256, 256)
        print(self.astar_map.dtypelatent)
        self.plt_fig, self.plt_obj = plotImage(self.image_size, self.plt_fig, self.plt_obj, self.old_depth_img, self.old_depth_img, self.astar_map)
        pixel_value = self.astar_map[self.current_pixel_coordinates[1],self.current_pixel_coordinates[0]]
        if self.current_pixel_coordinates == self. last_pixel_coordinates:
            return -1
        if pixel_value == 0:
            return 0
        else: return pixel_value
