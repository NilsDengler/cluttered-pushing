import matplotlib.pyplot as plt
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../utils"))
print(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
import numpy as np
import cv2
import random
import transforms as tfs
import warnings
import matplotlib
import os
import pybullet_data
import glob
from lazy_theta_star import calculate_lazy_theta_star, floodfill
from custom_utils import euler_from_quat, quat_from_euler, get_config, load_model, create_cylinder, create_capsule,\
     create_sphere, sample_pos_in_env, set_pose, remove_body, WHITE, RED, get_point, get_quat
from collections import deque
    # TURTLEBOT_URDF, joints_from_names, \
    # set_joint_positions, HideOutput, get_bodies, sample_placement, pairwise_collision, \
    # set_point, Point, create_box, stable_z, TAN, GREY, connect, PI, OrderedSet, \
    # wait_if_gui, dump_body, set_all_color, BLUE, child_link_from_joint, link_from_name, draw_pose, Pose, pose_from_pose2d, \
    # get_random_seed, get_numpy_seed, set_random_seed, set_numpy_seed, plan_joint_motion, plan_nonholonomic_motion, \
    # joint_from_name, safe_zip, draw_base_limits, BodySaver, WorldSaver, LockRenderer, elapsed_time, disconnect, flatten, \
    # INF, wait_for_duration, get_unbuffered_aabb, draw_aabb, DEFAULT_AABB_BUFFER, get_link_pose, get_joint_positions, \
    # get_subtree_aabb, get_pairs, get_distance_fn, get_aabb, set_all_static, step_simulation, get_bodies_in_region, \
    # AABB, update_scene, Profiler, pairwise_link_collision, BASE_LINK, get_collision_data, draw_pose2d, \
    # normalize_interval, wrap_angle, CIRCULAR_LIMITS, wrap_interval, Euler, rescale_interval, adjust_path, WHITE, RED, \
    # sample_pos_in_env, remove_body, get_euler, get_point, get_config, reset_sim, set_pose, get_quat,euler_from_quat, \
    # quat_from_euler, pixel_from_point, create_cylinder, create_capsule, create_sphere

class pushUtils:
    def __init__(self, sim, env_utils, p_id=0):
        self.sim = sim
        self.env_utils = env_utils
        self._p = p_id
        self.client_id = self.sim.client_id


    def include_curriculum_learning(self):
        if self.sim.change_curriculum_difficulty:
            if self.sim.curriculum_difficulty < self.sim.max_curriculum_iteration_steps - 1:
                self.sim.curriculum_difficulty += 1
            self.sim.change_curriculum_difficulty = False
        self.sim.curric.close_to_far_curriculum_fixed_obstacles(self.sim.curriculum_difficulty)
