import numpy as np
import cv2
import os
from custom_utils import get_config, euler_from_quat
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
        self.current_arm_trajectory = []
        self.current_initial_image = None
        self.global_count = 0
        self.current_collisions = []
        self.collision_list = []
        self.current_contacts = []
        self.contact_list = []
        self.shortest_path_length_list = []
        self.path_length_list = []


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

    def write_evaluation(self, is_RL=True):
        if self.sim.save_evaluations:
            os.makedirs(self.sim.evaluation_save_path, exist_ok=True)
            self.utils.save_np(np.asarray(self.success_list), self.sim.evaluation_save_path + "success_list.npy")
            if is_RL:
                self.utils.save_np(np.asarray(self.reward_list), self.sim.evaluation_save_path + "reward_list.npy")
                self.utils.save_np(np.asarray(self.run_length_list), self.sim.evaluation_save_path + "run_length_list.npy")
            self.utils.save_np(np.asarray(self.collision_list), self.sim.evaluation_save_path + "collision_list.npy")
            self.utils.save_np(np.asarray(self.contact_list), self.sim.evaluation_save_path + "contact_list.npy")
            self.utils.save_np(np.asarray(self.path_length_list), self.sim.evaluation_save_path + "path_length_list.npy")
            self.utils.save_np(np.asarray(self.shortest_path_length_list), self.sim.evaluation_save_path + "shortest_path_length_list.npy")


    def collect_evaluation_data(self, current_pose, last_pose, current_arm, last_arm, obj_contact, collision):
        #calculate current path length
        self.current_path_length += np.linalg.norm(last_pose - current_pose)
        #add position pairs to current path
        self.current_path.append([current_pose, last_pose])
        self.current_arm_trajectory.append([current_arm, last_arm])
        #add collison and contact bool
        if self.sim.first_object_contact == True:
            self.current_contacts.append(obj_contact)
        self.current_collisions.append(collision)


    def calculate_evaluation_data_RL(self, reached, reward, current_steps, current_pose):
        #save success
        if reached: self.success_list.append(True)
        else: self.success_list.append(False)
        #save reward
        self.reward_list.append(reward)
        #save current steps
        self.run_length_list.append(current_steps)
        #save contact and collision count of current episode
        if len(self.current_contacts) == 0:
            self.contact_list.append(0)
        else: self.contact_list.append(np.mean(self.current_contacts))
        if len(self.current_collisions) == 0:
            self.collision_list.append(0)
        else: self.collision_list.append(np.mean(self.current_collisions))
        #add goal pose to path of current episode
        self.current_path.append([current_pose, self.sim.goal_obj_conf[0]])
        self.current_path_length += np.linalg.norm(current_pose - self.sim.goal_obj_conf[0])
        #add current path length to list
        self.path_length_list.append(self.current_path_length)
        # add initial path length of current episode to list
        self.shortest_path_length_list.append(self.shortest_path_to_3D(self.sim.initial_shortest_path))
        # generate image of current episode
        self.straight_pushing_picture_RL()
        #reset all lists and values
        self.global_count += 1
        self.current_path_length = 0
        self.current_arm_trajectory.clear()
        self.current_path.clear()
        self.current_contacts.clear()
        self.current_collisions.clear()

    def straight_pushing_picture_RL(self):
        img = 255 - self.sim.initial_path_img.copy()
        start = np.array(self.cam.get_pos_in_image(self.current_path[0][0]))
        end = np.array(self.cam.get_pos_in_image(self.sim.goal_obj_conf[0]))
        img = cv2.rectangle(img, tuple(start - 8), tuple(start + 8), (0, 0, 255), -1)
        img = cv2.rectangle(img, tuple(end - 8), tuple(end + 8), (0, 255, 0), -1)
        self.line_drawing_RL(self.current_arm_trajectory, img, [0, 0, 0])
        self.line_drawing_RL(self.current_path, img, [255, 0, 0])
        cv2.imwrite(self.evaluation_save_path + "path_img_" + str(self.global_count) + ".png", img)


    def line_drawing_RL(self, path, img, color, size=2):
        for p in path:
            p_1 = self.cam.get_pos_in_image(p[0])
            p_2 = self.cam.get_pos_in_image(p[1])
            img = cv2.line(img, (p_2[0], p_2[1]), (p_1[0], p_1[1]), color, size)
        return img

    def shortest_path_to_3D(self, path):
        shortest_path_3D = []
        current_shortest_path_length = 0
        for idx in range(0, len(path)):
            world_points = self.cam.process_pixel_to_3dpoint(path[idx], self.sim.initial_true_depth, self.sim._workspace_bounds)
            shortest_path_3D.append(world_points)
        for idx in range(0, len(shortest_path_3D) - 1):
            current_shortest_path_length += np.linalg.norm(shortest_path_3D[idx] - shortest_path_3D[idx+1])
        return current_shortest_path_length


    def calculate_evaluation_data_baseline(self, reached, current_pose, initial_shortest_path):
        # save success
        if reached: self.success_list.append(True)
        else: self.success_list.append(False)
        # save contact and collision count of current episode
        if len(self.current_contacts) == 0:
            self.contact_list.append(0)
        else: self.contact_list.append(np.mean(self.current_contacts))
        if len(self.current_collisions) == 0:
            self.collision_list.append(0)
        else: self.collision_list.append(np.mean(self.current_collisions))
        # add goal pose to path of current episode
        self.current_path.append([current_pose, self.sim.goal_obj_conf[0][:2]])
        self.current_path_length += np.linalg.norm(current_pose - self.sim.goal_obj_conf[0][:2])
        # add current path length to list
        self.path_length_list.append(self.current_path_length)
        # add initial path length of current episode to list
        self.shortest_path_length_list.append(self.shortest_path_to_3D_baseline(initial_shortest_path))
        # generate image of current episode
        self.straight_pushing_picture_baseline()
        # reset all lists and values
        self.global_count += 1
        self.current_path_length = 0
        self.current_arm_trajectory.clear()
        self.current_path.clear()
        self.current_contacts.clear()
        self.current_collisions.clear()


    def world_to_grid(self,coords):
        pos = np.asarray([0, 0], dtype=int)
        pos[1] = int((coords[0] + 0.452) * (256 - 1) / 0.907)  # int((coords[0] + 0.45)*255/0.9) #
        pos[0] = int((coords[1] + 1.05) * (256 - 1) / 0.903)  # int((coords[1] + 0.75)*127/0.45)#
        return pos

    def shortest_path_to_3D_baseline(self, path):
        shortest_path_3D = []
        current_shortest_path_length = 0
        for idx in range(0, len(path)):
            world_points = self.grid_to_world(path[idx])
            shortest_path_3D.append(world_points)
        for idx in range(0, len(shortest_path_3D) - 1):
            current_shortest_path_length += np.linalg.norm(shortest_path_3D[idx] - shortest_path_3D[idx+1])
        return current_shortest_path_length

    def grid_to_world(self, coords):
        "gets grid coordinates and returns center of grid cell in absolute coords"
        pos = np.asarray([0, 0], dtype=float)
        pos[0] = coords[0]/(256 - 1)*0.907 - 0.452
        pos[1] = coords[1]/(256 - 1)*0.903 - 1.05
        return pos


    def line_drawing_baseline(self, path, img, color, size=2):
        for p in path:
            p_1 = self.world_to_grid(p[0])
            p_2 = self.world_to_grid(p[1])
            img = cv2.line(img, (p_2[0], p_2[1]), (p_1[0], p_1[1]), color, size)
        return img

    def straight_pushing_picture_baseline(self):
        img = self.sim.baseline.initial_path_img.copy()
        self.line_drawing_baseline(self.current_arm_trajectory, img, [0,0,0])
        self.line_drawing_baseline(self.current_path, img, [255,0,0])
        cv2.imwrite(self.sim.evaluation_save_path + "path_img_"+ str(self.global_count) + ".png", img)