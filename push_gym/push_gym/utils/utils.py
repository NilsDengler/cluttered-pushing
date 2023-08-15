import matplotlib.pyplot as plt
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../utils"))
print(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
import numpy as np
import cv2
import random
import warnings
import matplotlib
import os
import pybullet_data
import glob
from lazy_theta_star import floodfill
from custom_utils import euler_from_quat, quat_from_euler, get_config, load_model, create_cylinder, create_capsule,\
     create_sphere, sample_pos_in_env, set_pose, remove_body, WHITE, RED, get_point, get_quat

BLOCK_URDF_PATH = '../robots/assets/block/block.urdf'

class Utils:
    def __init__(self, sim, p_id=0):
        self.sim = sim
        self._p = p_id
        self.client_id = self.sim.client_id
        self.x_vec = -1
        self.y_vec = -1
        self.x_angle = -1
        self.y_angle = -1
        self.single_vec = -1
        self.x_id = -1
        self.y_id = -1
        self.z_id = -1
        self.obst_diameters = []
        self.obj_config = "random_obj"
        self.goal_config = "random_goal"
        self.ws_bounds = self.sim._workspace_bounds

    ######################################
    '''Camera functions'''
    ######################################
    def define_camera_parameters(self, width, height, fov, near, far):
        self.width = width
        self.height = height
        self.image_size = width
        self.fov = fov
        self.focal_length = (float(self.width) / 2) / np.tan((np.pi * self.fov / 180) / 2)
        self.aspect = self.width / self.height
        self.near = near
        self.far = far

    def define_camera(self, cyaw=None, cpitch=None, croll=None, cdist=None, eye_pos=None, target_pos=None,
                      up_vector=None, fov=None, near=None, far=None, with_rpy=True):
        if fov!=None: self.fov = fov
        if near!=None: self.near = near
        if far!=None: self.far = far
        if with_rpy:
            self.view_matrix = np.asarray(self._p.computeViewMatrixFromYawPitchRoll(target_pos, cdist, cyaw,
                                                                                      cpitch, croll, 2))
        else:
            self.view_matrix = np.asarray(self._p.computeViewMatrix(eye_pos, target_pos, up_vector))
        self.projection_matrix = self._p.computeProjectionMatrixFOV(self.fov, self.aspect, self.near, self.far)
        self.intrinsic_matrix = np.array([[self.focal_length, 0, float(self.width) / 2],
                                          [0, self.focal_length, float(self.height) / 2],
                                          [0, 0, 1]])

    def get_image(self):
        images = self._p.getCameraImage(self.width, self.height, self.view_matrix, self.projection_matrix,
                                          shadow=False, renderer=self._p.ER_BULLET_HARDWARE_OPENGL)

        rgb = np.array(images[2])[:,:,:3]
        depth = images[3]
        normalized_depth = depth*255
        normalized_depth = normalized_depth.astype(np.uint8)
        true_depth = self.far * self.near / (self.far - (self.far - self.near) * depth)
        return rgb, normalized_depth, true_depth

    def get_point_cloud(self, depth_img):
        fx = fy = 1 / np.tan(self.fov / 2)
        point_cloud = np.zeros((self.height, self.width, 3), dtype=np.float32)
        self.saved_transformation = np.zeros((self.height, self.width), dtype=np.float32)
        for h in range(self.height):
            for w in range(self.width):
                if depth_img[h][w] != 1:
                    x = (w - (self.width - 1) / 2 + 1) / (self.width - 1)
                    y = ((self.height - 1) / 2 - h) / (self.height - 1)
                    z = depth_img[h][w]
                    point_cloud[h, w, :] = [(x * z / fx), (y * z / fy), -z]
        return point_cloud

    def twoD_to_threeD(self, point, z):
        u0 = self.intrinsic_matrix[0, 2]
        v0 = self.intrinsic_matrix[1, 2]
        fx = self.intrinsic_matrix[0, 0]
        fy = self.intrinsic_matrix[1, 1]
        x = (point[0] - u0) * z / fx
        y = (point[1] - v0) * z / fy
        return [x,y, z]

    def process_pixel_to_3dpoint(self, p, current_true_depth, workspace_bounds):
        z = current_true_depth[p[1]][p[0]]
        unprocessed_point = self.twoD_to_threeD(p, z)
        camera_pose = self.get_cam_world_pose_from_view_matrix(self.view_matrix)
        world_points = self.transform_3Dpoint(camera_pose, np.array(unprocessed_point))
        world_points[2] = workspace_bounds[2][0]
        return world_points

    def save_rgb(self, img, path):
        cv2.imwrite(path, cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2BGR))

    def load_rgb(self, path):
        return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

    def show_rgb(self, img):
        cv2.cvtColor(img.astype(np.float32), cv2.COLOR_RGB2BGR)
        cv2.imshow("img", img)
        cv2.waitKey(0)

    def save_np(self, img, path):
        np.save(path, img)

    def draw_circle(self, img_array, pos):
        cv_img = img_array
        cv2.circle(np.float32(cv_img), (pos[0], pos[1]), 20, (203, 192, 255), -1)
        return cv_img

    def save_point_cloud(self, pc, path):
        np.save(path, pc)

    def load_point_cloud(self, path):
        return np.load(path)

    def get_pos_in_image(self, pos):
        obj_pos_in_cam = self.transform_world_point_to_camera_space(pos, self.view_matrix)
        pixel_points = np.asarray(self.camera_to_image_point(self.intrinsic_matrix, obj_pos_in_cam))
        pixel_points[0, 0] = self.image_size - pixel_points[0, 0]
        pixel_points = np.clip(pixel_points[0,:],0,255)
        return list(pixel_points)

    def process_images(self, image):
        unify = image.astype(np.float32) / 255.
        current_min, current_max = np.amin(unify), np.amax(unify)
        if current_max == current_min:
            return unify*0
        normed_min, normed_max = 0, 1
        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            try:
                x_normed = (unify - current_min) / (current_max - current_min)
            except Warning as e:
                print('error found:', e)
                print(unify, current_min, current_max, current_min)
                matplotlib.image.imsave("/home/user/dengler/Documents/robot_arm_RL/pushing_net/push_gym/push_gym/error_data/error.png", image)
                np.save("/home/user/dengler/Documents/robot_arm_RL/pushing_net/push_gym/push_gym/error_data/error.npy", image)
        x_normed = (unify - current_min) / (current_max - current_min)
        x_normed = x_normed * (normed_max - normed_min) + normed_min
        return x_normed

    def pixel_boundary_check(self, x, y, w, h):
        if (x < 0): x = 0
        elif(x > w - 2 ): x = w - 2
        if (y < 0): y=0
        elif (y > h - 2): y = h - 2
        return (x,y)

    def image_oob_check(self, x, y, window_size):
        oob_check_x = x + window_size > self.image_size or x - window_size < 0
        oob_check_y = y + window_size > self.image_size or y - window_size < 0
        return (oob_check_x or oob_check_y)

    def process_initial_image(self, img, obj_conf):
        obj_pixel_pos = self.get_pos_in_image(obj_conf[0])
        obj_pixel_pos = self.pixel_boundary_check(obj_pixel_pos[0], obj_pixel_pos[1], self.image_size,self.image_size)
        # cut arm from image and set it to 0 and 1
        depth_wo_arm = img.copy()
        depth_wo_arm[depth_wo_arm <= 218] = 255
        depth_wo_arm[depth_wo_arm < 220] = 1
        depth_wo_arm[depth_wo_arm >= 220] = 0
        kernel_dilate = (np.ones((7, 7), np.float32)) / 49
        kernel_erode = (np.ones((3, 3), np.float32)) / 9
        depth_only_obstacles = cv2.erode(depth_wo_arm.copy(), kernel_erode, iterations=2)
        depth_only_obstacles = cv2.dilate(depth_only_obstacles, kernel_dilate, iterations=3)
        depth_only_obstacles = floodfill(depth_only_obstacles.copy(), obj_pixel_pos)
        return depth_only_obstacles

    ######################################
    '''Transforms'''
    ######################################

    def rotate_image_and_get_point(self, image, obj_config, with_object_ori=False, with_subgoal_ori=False):
        """
        Rotates an OpenCV 2 / NumPy image about it's centre by the given angle
        (in degrees). The returned image will be large enough to hold the entire
        new image, with a black background
        """
        # get angle
        #with object orientation
        old_object_center = self.get_pos_in_image(obj_config[0])
        if with_object_ori:
            angle = -np.rad2deg(euler_from_quat(obj_config[1])[2])
        elif with_subgoal_ori:
            p0 = old_object_center
            p1 = self.get_pos_in_image(self.sim.sub_goal_queue[0][0])
            angle = -np.rad2deg((np.arctan2(*p1[::-1]) - np.arctan2(*p0[::-1])) )#% (2 * np.pi))
        else: angle = 0
        # Get the image size
        # No that's not an error - NumPy stores image matricies backwards
        image_size = (image.shape[1], image.shape[0])
        # Convert the OpenCV 3x2 rotation matrix to 3x3
        rot_mat = np.vstack(
            [cv2.getRotationMatrix2D(tuple(np.array(image_size) / 2), angle, 1.0), [0, 0, 1]]
        )

        rot_mat_notranslate = np.matrix(rot_mat[0:2, 0:2])
        # Shorthand for below calcs
        image_w2 = image_size[0] * 0.5
        image_h2 = image_size[1] * 0.5

        rotated_coords = [
            (np.array([-image_w2, image_h2]) * rot_mat_notranslate).A[0],
            (np.array([image_w2, image_h2]) * rot_mat_notranslate).A[0],
            (np.array([-image_w2, -image_h2]) * rot_mat_notranslate).A[0],
            (np.array([image_w2, -image_h2]) * rot_mat_notranslate).A[0]
        ]

        # Find the size of the new image
        x_coords = [pt[0] for pt in rotated_coords]
        x_pos = [x for x in x_coords if x > 0]
        x_neg = [x for x in x_coords if x < 0]

        y_coords = [pt[1] for pt in rotated_coords]
        y_pos = [y for y in y_coords if y > 0]
        y_neg = [y for y in y_coords if y < 0]

        right_bound, left_bound, top_bound, bot_bound = max(x_pos), min(x_neg), max(y_pos), min(y_neg)

        new_w = int(abs(right_bound - left_bound + image_size[0]))
        new_h = int(abs(top_bound - bot_bound + image_size[0]))

        # We require a translation matrix to keep the image centred
        trans_mat = np.matrix([
            [1, 0, int(new_w * 0.5 - image_w2)],
            [0, 1, int(new_h * 0.5 - image_h2)],
            [0, 0, 1]
        ])

        # Compute the tranform for the combined rotation and translation
        self.affine_mat = (np.matrix(trans_mat) * np.matrix(rot_mat))[0:2, :]
        extended_affine_mat = np.vstack([self.affine_mat, [0, 0, 1]])
        inv_affine_mat = np.linalg.inv(extended_affine_mat)
        # Apply the transform
        result = cv2.warpAffine(image,self.affine_mat,(new_w, new_h),flags=cv2.INTER_LINEAR)
        extended_object_center = np.append(old_object_center, [1])
        new_oc = np.matmul(self.affine_mat, extended_object_center).astype(np.int32)

        cp_1 = [new_oc[0,0] - 32, new_oc[0,1] - 32 ]
        cp_2 = [new_oc[0,0] + 32, new_oc[0,1] - 32 ]
        cp_3 = [new_oc[0,0] + 32, new_oc[0,1] + 32 ]
        cp_4 = [new_oc[0,0] - 32, new_oc[0,1] + 32 ]
        cp_1 = np.matmul(inv_affine_mat,  np.append(cp_1, [1])).astype(np.int32).tolist()[0]
        cp_2 = np.matmul(inv_affine_mat,  np.append(cp_2, [1])).astype(np.int32).tolist()[0]
        cp_3 = np.matmul(inv_affine_mat,  np.append(cp_3, [1])).astype(np.int32).tolist()[0]
        cp_4 = np.matmul(inv_affine_mat,  np.append(cp_4, [1])).astype(np.int32).tolist()[0]
        return result, new_oc, [cp_1, cp_2, cp_3, cp_4]

    def get_local_window(self, depth_img, obj_config, local_window_size):
        test_rotated_image, self.new_center, frame_corners = self.rotate_image_and_get_point(depth_img, obj_config, with_object_ori=True)
        local_window_size = int(local_window_size / 2)
        cropped_image = test_rotated_image[self.new_center[0, 1] - local_window_size: self.new_center[0, 1] + local_window_size,
                                           self.new_center[0, 0] - local_window_size: self.new_center[0, 0] + local_window_size]
        cropped_image = cv2.circle(cropped_image, (32, 32), 14, 221, -1)
        cropped_image[cropped_image <= 218] = 221
        cropped_image[cropped_image == 220] = 221
        if cropped_image.shape[1] != 64:
            cv2.imwrite(os.path.join(os.path.dirname(__file__), "../error_orig.png"), depth_img)
            cv2.imwrite(os.path.join(os.path.dirname(__file__), "../error_rotation.png"), test_rotated_image)
            cv2.imwrite(os.path.join(os.path.dirname(__file__), "../error_cropped.png"), cropped_image)
            assert False, "something went wrong in cropping the image"
        kernel = np.ones((3, 3), np.uint8)
        thresh = 1 - cv2.threshold(cropped_image.copy(), 219, 1, cv2.THRESH_BINARY)[1]
        denoised_image = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        denoised_image[denoised_image == 1] = 219
        denoised_image[denoised_image == 0] = 221
        return denoised_image, frame_corners

    def get_transformed_point(self, point, rot_matrix):
        extended_object_center = np.append(point, [1])
        return np.matmul(rot_matrix, extended_object_center).astype(np.int32)

    def ee_is_in_lw(self):
        pixel_points = self.get_pos_in_image(self.sim.tool_tip_pose[0])
        transformed_pixel = self.get_transformed_point(pixel_points, self.affine_mat)
        if transformed_pixel[0,0] > self.new_center[0, 0] - 32 \
                and transformed_pixel[0,1] > self.new_center[0, 1] - 32 \
                and transformed_pixel[0,0] < self.new_center[0, 0] + 32 \
                and transformed_pixel[0,1] < self.new_center[0, 1] + 32:
            return True
        else: return False

    def transform_world_point_to_camera_space(self, point, view_matrix):
        ps_homogeneous = np.append(point, 1.)
        ps_transformed = np.dot(np.array(view_matrix).reshape(4, 4).T, ps_homogeneous.T).T
        return ps_transformed[:3]

    def camera_to_image_points(self, intrinsics, camera_points):
        u0 = intrinsics[0, 2]
        v0 = intrinsics[1, 2]
        fx = intrinsics[0, 0]
        fy = intrinsics[1, 1]

        image_coordinates = np.empty((camera_points.shape[0], 2), dtype=np.int64)
        for i in range(camera_points.shape[0]):
            image_coordinates[i, 0] = int(np.round((camera_points[i, 0] * fx / camera_points[i, 2]) + u0))
            image_coordinates[i, 1] = int(np.round((camera_points[i, 1] * fy / camera_points[i, 2]) + v0))

        return image_coordinates

    def camera_to_image_point(self, intrinsics, camera_points):
        u0 = intrinsics[0, 2]
        v0 = intrinsics[1, 2]
        fx = intrinsics[0, 0]
        fy = intrinsics[1, 1]

        image_coordinates = np.empty((1, 2), dtype=np.int64)
        image_coordinates[0, 0] = int(np.round((camera_points[0] * fx / camera_points[2]) + u0))
        image_coordinates[0, 1] = int(np.round((camera_points[1] * fy / camera_points[2]) + v0))
        return image_coordinates

    def transform_3Dpoint(self, t, point):
        ps_homogeneous = np.append(point, 1.)
        ps_transformed = np.dot(t, ps_homogeneous.T).T
        return ps_transformed[:3]

    def get_cam_world_pose_from_view_matrix(self, view_matrix):
        cam_pose = np.linalg.inv(np.array(view_matrix).reshape(4, 4).T)
        cam_pose[:, 1:3] = -cam_pose[:, 1:3]
        return cam_pose



    ######################################
    '''Collision functions'''
    ######################################

    def check_object_generation_collision(self, id, pushing_obj=False, goal=False, count=0, sample_id=0):
        self.sim._p.performCollisionDetection()
        collision = []
        closest_check = []

        for o_id in self.sim.obstacles + [self.sim.pushing_object_id]:
            if o_id == id or o_id < 0: continue
            # check if object or goal is to close to obstacle
            if pushing_obj or goal:
                closest = self._p.getClosestPoints(bodyA=id, bodyB=o_id, distance=0.06,
                                                   physicsClientId=self.sim.client_id)
                if closest:
                    if count > 200:
                        if goal:
                            print(len(closest))
                            print(closest[0][8])
                            current_pos = np.asarray(get_config(id, self._p, self.sim.client_id)[0][:2])
                            print("collision loop for ", "object ", pushing_obj, "in environment: ", sample_id,
                                  " and is goal ", goal, " obj id: ", o_id)
                            print("goal coords: ", current_pos)
                            print("obst coords: ", get_config(o_id, self._p, self.sim.client_id))
                            print("distance: ", round(np.linalg.norm(
                                current_pos - np.asarray(get_config(o_id, self._p, self.sim.client_id)[0][:2])), 3))
                    return True
            collision, closest_check = self.collision_checks(collision, closest_check, id, o_id)
        if not pushing_obj and not goal:
            current_config = get_config(id, self._p, self.sim.client_id)
            goal_dist = round(
                np.linalg.norm(np.asarray(self.sim.goal_obj_conf[0][:2]) - np.asarray(current_config[0][:2])), 3)
            return True if collision or closest_check or goal_dist < 0.2 else False
        return True if collision or closest_check else False

    def check_pushing_object_collision(self, obj_id, pos, distance=0.05):
        self.sim.builder.reset_obj_fix(obj_id, pos)
        return self.check_object_generation_collision(obj_id, True)

    def check_for_collision_EE(self):
        collision = []
        closest_check = []
        # check if any obstacle on the table is coliding with any link of arm or gripper
        for env_obj in self.sim.obstacles + [self.sim.table]:
            for i in self.sim._robot_joint_indices:
                collision, closest_check = self.collision_checks(collision, closest_check, self.sim.robot_arm_id,
                                                                 env_obj, link_A=i)
            if not self.sim.with_rod:
                for j in self.sim._gripper_joint_indices:
                    if self.sim._gripper_id:
                        collision, closest_check = self.collision_checks(collision, closest_check, self.sim._gripper_id,
                                                                         env_obj, link_A=j)
                    else:
                        collision, closest_check = self.collision_checks(collision, closest_check,
                                                                         self.sim.robot_arm_id, env_obj, link_A=j)
            else:
                collision, closest_check = self.collision_checks(collision, closest_check, self.sim.robot_arm_id,
                                                                 env_obj, link_A=self.sim.rod_index)
        return True if collision or closest_check else False

    def check_for_collision_object(self):
        collision = []
        closest_check = []
        # check if any obstacle on the table is coliding with the object
        for env_obj in self.sim.obstacles:
            collision, closest_check = self.collision_checks(collision, closest_check, self.sim.pushing_object_id,
                                                             env_obj)
        return True if collision or closest_check else False

    def collision_checks(self, collision, closest_check, body_A, body_B, link_A=-1, link_B=-1):
        collision += self._p.getContactPoints(bodyA=body_A, bodyB=body_B, linkIndexA=link_A, linkIndexB=link_B,
                                              physicsClientId=self.sim.client_id)
        closest_check += self._p.getClosestPoints(bodyA=body_A, bodyB=body_B, linkIndexA=link_A, linkIndexB=link_B,
                                                  distance=0.001, physicsClientId=self.sim.client_id)
        return collision, closest_check

    def object_contact(self, dist=0.001):
        obj_contact = []
        if not self.sim.with_rod:
            for j in self.sim._gripper_joint_indices:
                if self.sim._gripper_id:
                    obj_contact += self._p.getClosestPoints(bodyA=self.sim._gripper_id,
                                                            bodyB=self.sim.pushing_object_id, linkIndexA=j,
                                                            distance=dist, physicsClientId=self.sim.client_id)
                else:
                    obj_contact += self._p.getClosestPoints(bodyA=self.sim.robot_arm_id,
                                                            bodyB=self.sim.pushing_object_id, linkIndexA=j,
                                                            distance=dist, physicsClientId=self.sim.client_id)
        else:
            obj_contact += self._p.getClosestPoints(bodyA=self.sim.robot_arm_id, bodyB=self.sim.pushing_object_id,
                                                    linkIndexA=self.sim.rod_index, distance=dist,
                                                    physicsClientId=self.sim.client_id)
        return True if obj_contact else False

    ######################################
    '''Environment building functions'''
    ######################################

    def get_cylinder_obj(self):
        obj_id = create_cylinder(self._p, self.client_id, 0.03, 0.06, mass=.1, color=RED)
        self._p.changeDynamics(obj_id, -1, restitution=0.0, lateralFriction=1.0, spinningFriction=1.0, rollingFriction=0.0001)
        return obj_id

    def get_capsule_obj(self):
        obj_id = create_capsule(self._p, self.client_id, 0.03, 0.06, mass=.1, color=RED)
        self._p.changeDynamics(obj_id, -1, restitution=0.0, lateralFriction=1.0, spinningFriction=1.0, rollingFriction=0.0001)
        return obj_id

    def get_sphere_obj(self):
        obj_id = create_sphere(self._p, self.client_id, 0.03, mass=.1, color=RED)
        self._p.changeDynamics(obj_id, -1, restitution=0.0, lateralFriction=1.0, spinningFriction=1.0, rollingFriction=0.0001)
        return obj_id

    def get_block_obj(self):
        xpos, ypos, zpos, yaw = sample_pos_in_env(self.sim._workspace_bounds, self._p)
        obj_id = self._p.loadURDF(os.path.join(os.path.dirname(__file__), BLOCK_URDF_PATH),
                                  basePosition=[xpos, ypos, zpos], baseOrientation=list(yaw),
                                  physicsClientId=self.client_id)
        self._p.changeDynamics(obj_id, -1, mass=1, lateralFriction=.5, rollingFriction=0.0001)
        return obj_id

    def get_cube_obj(self):
        xpos, ypos, zpos, yaw = sample_pos_in_env(self.sim._workspace_bounds, self._p)
        obj_name = "cube_small.urdf"
        #obj_name = "lego/lego.urdf"
        #obj_name = "objects/mug.urdf"
        #obj_name = "duck_vhacd.urdf"
        globalScaling = 1
        if "mug" in obj_name:
            globalScaling = 0.75
        obj_id = self._p.loadURDF(os.path.join(pybullet_data.getDataPath(), obj_name),
                                  basePosition=[xpos, ypos, zpos], baseOrientation=list(yaw),
                                  flags=self._p.URDF_USE_MATERIAL_COLORS_FROM_MTL,
                                  physicsClientId=self.client_id,globalScaling=globalScaling)

        self._p.changeDynamics(obj_id, -1, restitution=0.0, lateralFriction=1.0, spinningFriction=1.0,
                               rollingFriction=0.0001)
        self._p.changeVisualShape(obj_id, -1, rgbaColor=[1, 0, 0, 1], physicsClientId=self.client_id)
        return obj_id

    def generate_obj(self):
        if "static" in self.obj_config:
            return self.generate_static_obj()
        elif "random" in self.obj_config:
            return self.generate_random_pushing_object()
        else:
            raise AssertionError("wrong obj_config")

    def generate_random_pushing_object(self):
        #obj_id = self.get_cylinder_obj()
        #obj_id = self.get_sphere_obj()
        #obj_id = self.get_capsule_obj()
        #obj_id = self.get_cube_obj()
        obj_id = self.get_block_obj()
        current_config = get_config(obj_id, self._p, self.client_id)
        if self.check_object_generation_collision(obj_id, pushing_obj=True):
            self.reset_obj(obj_id)
        return obj_id

    def reset_obj(self, id, positons=None):
        if positons:
            self.reset_obj_fix()
            return
        if "random" in self.obj_config:
                self.reset_obj_random(id)
        if "static" in self.obj_config:
            set_pose(id, self.sim.init_obj_conf, self._p, self.client_id)

    def reset_obj_random(self, id):
        collision = True
        obj_config = get_config(id, self._p, self.client_id)
        count = 0
        while collision:
            xpos, ypos, zpos, yaw = sample_pos_in_env(self.sim._workspace_bounds, self._p)
            #if np.linalg.norm(np.array(obj_config[0]) - np.array([xpos, ypos, zpos])) < 0.2:
            #        continue
            set_pose(id, ([xpos, ypos, zpos], list(yaw)), self._p,self.client_id)
            #print(xpos, ypos, zpos)
            #self._p.stepSimulation(physicsClientId=self.client_id)
            #self.sim.step_simulation(self.sim.per_step_iterations)
            collision = self.check_object_generation_collision(id, pushing_obj=True, count=count)
            count += 1
        return


    def reset_obj_fix(self, id, positions):
        pos = positions[0]
        ori = positions[1]
        if len(positions) == 3:
            ori = quat_from_euler([0,0,positions[2]])
            pos = positions[0:2]+[self.sim._workspace_bounds[2,0]]
        set_pose(id, (pos, ori), self._p, self.client_id)
        self._p.stepSimulation(physicsClientId=self.client_id)
        #self.sim.step_simulation(self.sim.per_step_iterations)
        return

    def get_random_non_cubic_object(self):
        urdf_pattern = os.path.join(self.sim._urdfRoot, 'random_urdfs/*[1-9]/*.urdf')
        found_object_directories = glob.glob(urdf_pattern)
        return found_object_directories

    def generate_static_obj(self):
        # fixed pos
        if self.obj_config == "static_middle":
            xpos, ypos, zpos, yaw = 0, -0.3, 0.05, quat_from_euler([0, 0, 0])
        elif self.obj_config == "static_interpol_perc":
            # interpolated on percentage on line
            point = self.interpol_percentage_pos(np.array(self.sim.initial_arm_position[:2]), np.array((-0.4, -0.75)), 0.2)
            xpos, ypos, zpos, yaw = point[0], point[1], 0.05, quat_from_euler([0, 0, 0])
        # interpolated on line
        elif self.obj_config == "static_interpol_x":
            xpos, ypos, zpos, yaw = -0.15, self.interpolate_pos([-0.4, 0.0], [-0.75, -0.198],
                                                                -0.15), 0.05, quat_from_euler([0, 0, 0])
        urdf_path = os.path.join(os.path.dirname(__file__), "../robots/assets/objects/cube_test.urdf")
        obj_id = self._p.loadURDF(os.path.join(pybullet_data.getDataPath(), 'cube_small' + ".urdf"),
                                  basePosition=[xpos, ypos, zpos], baseOrientation=list(yaw),
                                  flags=self._p.URDF_USE_MATERIAL_COLORS_FROM_MTL,
                                  physicsClientId=self.client_id)
        self._p.changeVisualShape(obj_id, -1, rgbaColor=[1, 0, 0, 1], physicsClientId=self.client_id)
        return obj_id

    def generate_obj_on_line(self,id, x_change, y_change):
        collision = True
        obj_config = get_config(id, self._p, self.client_id)
        while collision:
            yaw = 0
            if self.sim.with_ori:
                yaw = random.uniform(-np.pi, np.pi)
            xpos, ypos, zpos, ori = random.uniform(-x_change, x_change), random.uniform(-y_change, 0), 0.02, quat_from_euler([0, 0, yaw])
            if np.linalg.norm(np.array(obj_config[0]) - np.array([xpos, ypos, zpos])) < 0.2:
                continue
            set_pose(id, ([xpos, ypos, zpos], list(yaw)), self._p, self.client_id)
            self._p.stepSimulation(physicsClientId=self.client_id)
            #self.sim.step_simulation(self.sim.per_step_iterations)
            collision = self.check_object_generation_collision(id, pushing_obj=True)
        return

    ###########################
    '''Generate Goal functions'''
    ##########################

    def generate_goal(self, id):
        if self.goal_config == "goal_on_line":
            return self.generate_goal_on_line()
        elif self.goal_config == "static_goal_right":
            return self.generate_static_goal_right()
        elif self.goal_config == "static_goal_middle":
            return self.generate_static_goal_middle()
        elif self.goal_config == "random_goal":
            return self.generate_random_goal(id)
        else:
            raise AssertionError("wrong goal_config")

    def generate_random_goal(self, id, which_sample=0, min_distance=0, max_distance=1.2):
        obj_config = get_config(id, self._p, self.client_id)
        orig_pose = obj_config
        collision = True
        count = 0
        while collision:
            if which_sample==0: xpos, ypos, zpos, yaw = sample_pos_in_env(self.sim._workspace_bounds, self._p, z=obj_config[0][2])
            elif which_sample==1: xpos, ypos, zpos, yaw = self.sample_tg_pose(obj_config, max_distance,z=obj_config[0][2])
            elif which_sample == 2: xpos, ypos, zpos, yaw = self.sample_tg_pose_in_range(obj_config, min_distance, max_distance)
            if np.linalg.norm(np.array(obj_config[0][:2]) - np.array([xpos, ypos])) < 0.02:
                continue
            set_pose(id, ([xpos, ypos, zpos], list(yaw)), self._p, self.client_id)
            self._p.stepSimulation(physicsClientId=self.client_id)
            #self.sim.step_simulation(self.sim.per_step_iterations)
            collision = self.check_object_generation_collision(id, goal=True, count=count)
            count += 1
        goal_pose = get_config(id, self._p, self.client_id)
        self.reset_obj_fix(id, orig_pose)
        return goal_pose

    def generate_multiple_goals(self, id):
        final_points = []
        orig_pose = [list(get_point(id, self._p, self.client_id)), list(get_quat(id, self._p, self.client_id))]
        obj_config = get_config(id, self._p, self.client_id)
        to_close = True
        target_points = self.sample_tg_pose_in_circle(obj_config, 20, 0.20)
        for i in target_points:
            if not self.oob_check(i[0], i[1], i[2]):
                final_points.append(i)
        return final_points

    def generate_fixed_goal(self, position):
        xpos, ypos, zpos, yaw = position[0], position[1], 0.02, quat_from_euler([0, 0, position[2]])
        return [[xpos, ypos, zpos], list(yaw)]


    def generate_goal_on_line(self, y_change=False):
        yaw = 0
        y_pos_change = 0
        if self.sim.with_ori:
            yaw = random.uniform(-np.pi, np.pi)
        if y_change:
            y_pos_change = random.uniform(0.05, 0.35)
        xpos, ypos, zpos, ori = random.uniform(-0.4,0.4), self.sim._workspace_bounds[1][1]+y_pos_change, 0.02, quat_from_euler([0, 0, yaw])
        return [[xpos, ypos, zpos], list(ori)]

    ###########################
    '''Generate Obstacle functions'''
    ###########################
    def include_obstacles(self, obstacle_list, obstacle_num ):
        for n in range(obstacle_num):
            obst_id = self.generate_random_obstacles(True)
            obstacle_list.append(obst_id)
        return obstacle_list

    def remove_obstacles(self, obstacle_list):
        for id in obstacle_list:
            remove_body(id, self._p)

    def get_obstacle_pos(self, obstacle_list, target_pos, target_ori):
        obstacles_pos = []
        for id in obstacle_list:
            obst_conf = get_config(id, self._p, self.client_id)
            obst_pos_in_target, obst_orn_in_target = self.pb_transformation(target_pos, target_ori, obst_conf)
            obstacles_pos.append(list(obst_pos_in_target))
        return obstacles_pos


    def reset_obstacle_pose(self, id):
        collision = True
        obj_config = get_config(id, self._p, self.sim.client_id)
        while collision:
            xpos, ypos, zpos, yaw = sample_pos_in_env(self._workspace_bounds, self._p)
            #xpos, ypos, zpos, yaw = self.sample_tg_pose(obj_config, 0.15)
            if self.oob_check(xpos, ypos) or np.linalg.norm(np.array(obj_config[0]) - np.array([xpos, ypos, zpos])) < 0.1:
                continue
            set_pose(id, ([xpos, ypos, zpos], list(yaw)), self._p,self.sim.client_id)
            self.sim._p.stepSimulation(physicsClientId=self.sim.client_id)
            #self.sim.step_simulation(self.sim.per_step_iterations)
            collision = self.collision.check_object_generation_collision(id)
        return


    def generate_obstacle_in_bound_window(self, xpos=None):
        obstacle_names = ["large_cube_02.urdf"]#["large_cube_02.urdf", "large_cube_01.urdf", "large_cube_006.urdf"]
        if xpos is None:
            x_bounds = [[0.0, 0.2], [-0.2, 0.0]]
            left_or_right = random.randint(0, 1)
            xpos = random.uniform(x_bounds[left_or_right][0], x_bounds[left_or_right][1])
        ypos = random.uniform(self.sim._workspace_bounds[1, 0], self.sim._workspace_bounds[1, 1])
        zpos = self.sim._workspace_bounds[2, 0]
        yaw = self._p.getQuaternionFromEuler([0, 0, random.uniform(-np.pi, np.pi)])
        urdf_path = os.path.join(os.path.dirname(__file__), "../robots/assets/objects/",
                                 obstacle_names[random.randrange(len(obstacle_names))])
        obj_id = self._p.loadURDF(urdf_path, [xpos, ypos, zpos], list(yaw),
                                 physicsClientId=self.client_id)  # , useFixedBase=1)
        self._p.stepSimulation(physicsClientId=self.client_id)
        #self.sim.step_simulation(self.sim.per_step_iterations)
        return obj_id


    def generate_random_obstacles(self, cubic, num_obstacle_names=None):
        obstacle_names = ["large_cube_01.urdf", "large_cube_006.urdf"]
        if not cubic:
            obstacle_names += self.get_random_non_cubic_object()
        xpos, ypos, zpos, yaw = sample_pos_in_env(self.sim._workspace_bounds, self._p)
        zpos = zpos + 0.02  # for safety
        if num_obstacle_names:
            obstacle_name = obstacle_names[num_obstacle_names]
        else:
            obstacle_name = obstacle_names[random.randrange(len(obstacle_names))]
        urdf_path = os.path.join(os.path.dirname(__file__), "../robots/assets/objects/",obstacle_name)
        with open(urdf_path) as f:
            content = f.readlines()
            for line in content:
                if "box size" in line:
                    value = float(line.split('\"')[1].split(' ')[0])
                    self.obst_diameters.append(value)
                    break

        obj_id = self._p.loadURDF(urdf_path, [xpos, ypos, zpos], list(yaw), physicsClientId=self.client_id, useFixedBase=True)
        self._p.stepSimulation(physicsClientId=self.client_id)
        #self.sim.step_simulation(self.sim.per_step_iterations)
        if self.check_object_generation_collision(obj_id): self.reset_obstacle_pose(obj_id)
        return obj_id

    def generate_fixed_obstacle(self, positions, num_obstacle_names):
        obstacle_names = ["large_cube_06.urdf", "large_cube_05.urdf", "large_cube_04.urdf", "large_cube_03.urdf", "large_cube_02.urdf", "large_cube_01.urdf", "large_cube_006.urdf"]
        ori = quat_from_euler([0,0,positions[2]])
        pos = positions[0:2]+[0.025]
        urdf_path = os.path.join(os.path.dirname(__file__), "../robots/assets/objects/",
                                 obstacle_names[num_obstacle_names])
        with open(urdf_path) as f:
            content = f.readlines()
            for line in content:
                if "box size" in line:
                    value = float(line.split('\"')[1].split(' ')[0])
                    self.obst_diameters.append(value)
                    break

        obj_id = self._p.loadURDF(urdf_path, pos, ori, physicsClientId=self.client_id, useFixedBase=True)
        self._p.changeDynamics(obj_id, -1)#, mass=1e6)
        #self._p.stepSimulation(physicsClientId=self.client_id)
        return obj_id

    def choose_obstacle_build(self, case):
        if case == 0:
            #mid obstacle horizontal
            self.sim.obstacles.append(self.generate_fixed_obstacle([0, -0.475, 0], 1))
        if case == 1:
            #mid obstacle vertical
            self.sim.obstacles.append(self.generate_fixed_obstacle([0, -0.475, np.pi/2], 3))
        elif case == 2:
            #narrow gab obstacle
            self.sim.obstacles.append(self.generate_fixed_obstacle([-0.30, -0.475, 0], 2))
            self.sim.obstacles.append(self.generate_fixed_obstacle([0.30, -0.475, 0], 2))
        #elif case == 3:
            # zickzack
        #    self.sim.obstacles.append(self.generate_fixed_obstacle([-0.15, -0.575, 0], 0))
        #    self.sim.obstacles.append(self.generate_fixed_obstacle([0.15, -0.325, 0], 0))
        elif case == 4:
            # four points
            self.sim.obstacles.append(self.generate_fixed_obstacle([-0.15, -0.55, 0], 6))
            self.sim.obstacles.append(self.generate_fixed_obstacle([0.15, -0.55, 0], 6))
            self.sim.obstacles.append(self.generate_fixed_obstacle([0.15, -0.35, 0], 6))
            self.sim.obstacles.append(self.generate_fixed_obstacle([-0.15, -0.35, 0], 6))
        # hard
        if case == 5:
            #two mid obstacle horizontal
            self.sim.obstacles.append(self.generate_fixed_obstacle([0, -0.6, 0], 3))
            self.sim.obstacles.append(self.generate_fixed_obstacle([0, -0.3, 0], 3))
        #elif case == 6:
        #    # zickzack more close
        #    self.sim.obstacles.append(self.generate_fixed_obstacle([-0.15, -0.475, 0], 0))
        #    self.sim.obstacles.append(self.generate_fixed_obstacle([0.15, -0.325, 0], 0))
        elif case == 7:
            self.sim.obstacles.append(self.generate_fixed_obstacle([-0.1, -0.525, 0], 6))
            self.sim.obstacles.append(self.generate_fixed_obstacle([0.1, -0.525, 0], 6))
            self.sim.obstacles.append(self.generate_fixed_obstacle([0.1, -0.325, 0], 6))
            self.sim.obstacles.append(self.generate_fixed_obstacle([-0.1, -0.325, 0], 6))
        # very hard
        #elif case == 8:
            # zickzack more close and longer
        #    self.sim.obstacles.append(self.generate_fixed_obstacle([-0.15, -0.475, 0], 0))
        #    self.sim.obstacles.append(self.generate_fixed_obstacle([0.15, -0.375, 0], 0))
        elif case == 9:
            #changing mit obstacle
            size, angle = random.randint(1, 5), random.uniform(0, np.pi / 2)
            self.sim.obstacles.append(self.generate_fixed_obstacle([0, -0.475, angle], size))
        elif case == 10:
            for n in range(10):
                obst_id = self.generate_random_obstacles(True, 1)
                self.sim.obstacles.append(obst_id)
        elif case == 11:
            # zickzack more close and longer
            self.sim.obstacles.append(self.generate_fixed_obstacle([0.15, -0.475, 0], 5))
            self.sim.obstacles.append(self.generate_fixed_obstacle([0.25, -0.375, 0], 5))
            self.sim.obstacles.append(self.generate_fixed_obstacle([0.35, -0.275, 0], 5))
            self.sim.obstacles.append(self.generate_fixed_obstacle([0., -0.575, 0], 5))
            self.sim.obstacles.append(self.generate_fixed_obstacle([0.15, -0.675, 0], 5))
            self.sim.obstacles.append(self.generate_fixed_obstacle([0.25, -0.775, 0], 5))
            self.sim.obstacles.append(self.generate_fixed_obstacle([0.35, -0.875, 0], 5))

            self.sim.obstacles.append(self.generate_fixed_obstacle([-0.15, -0.475, 0], 5))
            self.sim.obstacles.append(self.generate_fixed_obstacle([-0.25, -0.375, 0], 5))
            self.sim.obstacles.append(self.generate_fixed_obstacle([-0.35, -0.275, 0], 5))
            self.sim.obstacles.append(self.generate_fixed_obstacle([-0.15, -0.675, 0], 5))
            self.sim.obstacles.append(self.generate_fixed_obstacle([-0.25, -0.775, 0], 5))
            self.sim.obstacles.append(self.generate_fixed_obstacle([-0.35, -0.875, 0], 5))
        elif case == 12:
            # zickzack more close and longer
            self.sim.obstacles.append(self.generate_fixed_obstacle([0.15, -0.475, 0], 5))
            self.sim.obstacles.append(self.generate_fixed_obstacle([0.25, -0.375, 0], 5))
            self.sim.obstacles.append(self.generate_fixed_obstacle([0.35, -0.275, 0], 5))
            self.sim.obstacles.append(self.generate_fixed_obstacle([0., -0.575, 0], 5))
            self.sim.obstacles.append(self.generate_fixed_obstacle([-0.15, -0.675, 0], 5))
            self.sim.obstacles.append(self.generate_fixed_obstacle([-0.25, -0.775, 0], 5))
            self.sim.obstacles.append(self.generate_fixed_obstacle([-0.35, -0.875, 0], 5))
            self.sim.obstacles.append(self.generate_fixed_obstacle([-0.15, -0.375, np.pi/4], 2))
            self.sim.obstacles.append(self.generate_fixed_obstacle([0.15, -0.775, np.pi / 4], 2))
    ###########################
    '''Math functions'''
    ###########################

    def oob_check(self, xpos, ypos):
        oob_check_x = xpos > self.sim._collision_bounds[0][0] and xpos < self.sim._collision_bounds[0][1]
        oob_check_y = ypos > self.sim._collision_bounds[1][0] and ypos < self.sim._collision_bounds[1][1]
        return not(oob_check_x or oob_check_y)

    def pose_in_circle(self, obj_pos, r, theta):
        return (obj_pos[0][0] + r * np.cos(theta), obj_pos[0][1] + r * np.sin(theta), self.sim._workspace_bounds[2][0])

    def sample_tg_pose_in_circle(self, obj_pos, num_points, r):
        circle_points = []
        slices = 2 * np.pi/num_points
        for i in range(num_points):
            theta = slices*i
            circle_points.append(self.pose_in_circle(obj_pos, r, theta))
        return circle_points

    def sample_tg_pose_in_range(self, obj_pos, low, high):
        bounds = self.ws_bounds
        valid = False
        steps = 0
        if obj_pos[0][0] < 0:
            if obj_pos[0][1] < -0.5:
                angle_low = 0
            else:
                angle_low = 3*np.pi/2
        else:
            if obj_pos[0][1] < -0.5:
                angle_low = np.pi/2
            else:
                angle_low = np.pi

        while not valid:
            distance = (high - low)*np.sqrt(np.random.random_sample()) + low
            angle = np.pi/2*np.random.random_sample() + angle_low
            target = [obj_pos[0][0] + distance*np.cos(angle), obj_pos[0][1] + distance*np.sin(angle), bounds[2][0]]
            if (target[0] >= bounds[0][0]) and (target[0] <= bounds[0][1]) and (target[1] <= bounds[1][0]) and (target[1] >= bounds[1][1]):
                valid = True
            steps += 1
        yaw = 2*np.pi*np.random.random_sample()
        yaw = self._p.getQuaternionFromEuler([0, 0, yaw])
        return target[0], target[1], target[2], yaw


    def sample_tg_pose(self, obj_pos, tg_pose_rnd_std, z=None):
        # get workspace limits
        Done = False
        while not Done:
            ws_lim = self.sim._workspace_bounds
            x_min, x_max, y_min, y_max = ws_lim[0][0], ws_lim[0][1], ws_lim[1][1],ws_lim[1][0]

            # Add a Gaussian noise to position
            mu, sigma = 0., tg_pose_rnd_std
            noise = np.random.normal(mu, sigma, 3)
            px = obj_pos[0][0] + noise[0]
            py = obj_pos[0][1] + noise[1]
            yaw = euler_from_quat(obj_pos[1])[2] + noise[2]
            px = np.clip(px, x_min, x_max)
            py = np.clip(py, y_min, y_max)
            pz = ws_lim[2][0]  # obj_pos[1][2]
            if z:
                pz = z
            yaw = self._p.getQuaternionFromEuler([0, 0, yaw])
            Done = True
            if np.linalg.norm(np.array(obj_pos[0][:2]) - np.array([px, py])) < 0.02:
                    Done = False

        return px, py, pz, yaw

    def interpolate_pos(self, p1, p2, x_new):
        y_new = np.interp(x_new, p1, p2)
        return y_new

    def interpol_percentage_pos(self, p1, p2, perc):
        return p1+perc*(p2-p1)


    def get_point_on_unit_vector(self, P, Q):
        PQ = np.array(Q) - np.array(P)
        # create unit vector
        uv = PQ / np.linalg.norm(PQ)
        T, inv_T = self.get_transform(uv, Q)
        og_x = np.matmul(inv_T, np.array((0.15, 0, 1)))
        og_x = og_x[:2] / og_x[2]

        mu, sigma = 0, 0.02
        noise = np.random.normal(mu, sigma, 2)
        noisy_point = og_x + noise
        return noisy_point
        #return (np.array((1,0))*0.2)

    def get_random_point_in_cone(self, P, Q, angle):
        PQ = np.array(Q) - np.array(P)
        # create unit vector
        uv = PQ / np.linalg.norm(PQ)
        T, inv_T = self.get_transform(uv, Q)
        rand_angle = random.uniform(-angle, angle)
        rand_point = random.uniform(0.1,0.2)
        rand_vector = self.rotate(np.array((1, 0)), rand_angle)
        return_point = np.matmul(inv_T, np.concatenate((rand_vector * rand_point, np.ones(1)), axis=0))
        return_point = return_point[:2] / return_point[2]
        return (return_point)

    def check_ee_in_object_cone(self, P, Q, ee, angle):
        #P = target, Q = Object
        PQ = np.array(Q)-np.array(P)
        #create unit vector
        uv = PQ / np.linalg.norm(PQ)
        #create transformation matrix for Q
        T, inv_T = self.get_transform(uv, Q)
        transformed_ee = np.matmul(T,(np.array((ee[0], ee[1], 1))))[:2]#
        #create rotation vectors to create cone
        upper_vec = self.rotate(np.array((1,0)), angle)#*0.2
        lower_vec = self.rotate(np.array((1,0)), -angle)#*0.2

        #self.visualize_axis(inv_T)
        #self.visualize_angles(inv_T,upper_vec, lower_vec)
        return self.inside_cone(upper_vec, lower_vec, transformed_ee)

    def check_ee_in_object_cone_viz(self, P, Q, ee, angle):
        #P = target, Q = Object
        PQ = np.array(Q)-np.array(P)
        #create unit vector
        uv = PQ / np.linalg.norm(PQ)
        #create transformation matrix for Q
        T, inv_T = self.get_transform(uv, Q)
        transformed_ee = np.matmul(T,(np.array((ee[0], ee[1], 1))))[:2]#
        #create rotation vectors to create cone
        upper_vec = self.rotate(np.array((1,0)), angle)#*0.2
        lower_vec = self.rotate(np.array((1,0)), -angle)#*0.2

        #self.visualize_axis(inv_T)
        self.visualize_angles(inv_T,upper_vec, lower_vec)
        return self.inside_cone(upper_vec, lower_vec, transformed_ee)

    def get_shortest_path_direction(self, P, Q):
        if P == Q: return [0,0]
        PQ = np.array(Q)-np.array(P)
        #create unit vector
        uv = PQ / np.linalg.norm(PQ)
        #print(uv)
        return list(uv)

    def rotate(self, vector, angle):
        theta = np.deg2rad(angle)
        c, s = np.cos(theta), np.sin(theta)
        R = np.array(((c, -s), (s, c)))
        return R.dot(vector)

    def inside_cone(self, v_1, v_2, p):
        a = np.cross(v_1, p)
        b = np.cross(v_2, p)
        return True if (a < 0 and b > 0) else False

    def get_transform(self, uv, Q):
        theta = -np.arctan2(uv[1], uv[0])
        c, s = np.cos(theta), np.sin(theta)
        rot = np.array(((c, -s, 0),(s, c, 0),(0,0,1)))
        trans = np.array(((1, 0, -Q[0]),(0,1, -Q[1]),(0,0,1)))
        T = np.matmul(rot,trans)
        T_inv = np.linalg.inv(T)
        return T, T_inv

    def dist_normalized(self, c_min, c_max, dist, what="object"):
        normed_min, normed_max = 0, 1
        if (dist - c_min) == 0 or (c_max - c_min) == 0:
            print("dist: ", dist, "c_min: ", c_min, "c_max: ", c_max, "for: ", what)
        x_normed = (dist - c_min) / (c_max - c_min)
        x_normed = x_normed * (normed_max - normed_min) + normed_min
        return round(x_normed, 4)

    ###########################
    '''Pybullet Goal functions'''

    ##########################
    def pb_transformation(self, target_frame_pos, target_frame_ori, world_coords):
        return self._p.multiplyTransforms(target_frame_pos, target_frame_ori, world_coords[0], world_coords[1])

    ###########################
    '''Debug functions'''
    ###########################
    def visualize_axis(self, inv_T):
        og = np.matmul(inv_T, np.array((0, 0, 1)))
        og = og[:2] / og[2]
        og_x = np.matmul(inv_T, np.array((1, 0, 1)))
        og_x = og_x[:2] / og_x[2]
        og_y = np.matmul(inv_T, np.array((0, -1, 1)))
        og_y = og_y[:2] / og_y[2]

        if self.x_vec < 0:
            self.x_vec = self._p.addUserDebugLine([og[0], og[1], 0.02], [og_x[0], og_x[1], 0.02], [1, 0, 0],
                                                  physicsClientId=self.client_id)
            self.y_vec = self._p.addUserDebugLine([og[0], og[1], 0.02], [og_y[0], og_y[1], 0.02], [0, 1, 0],
                                                  physicsClientId=self.client_id)
        else:
            self._p.addUserDebugLine([og[0], og[1], 0.02], [og_x[0], og_x[1], 0.02], [1, 0, 0],
                                     replaceItemUniqueId=self.x_vec, physicsClientId=self.client_id)
            self._p.addUserDebugLine([og[0], og[1], 0.02], [og_y[0], og_y[1], 0.02], [0, 1, 0],
                                     replaceItemUniqueId=self.y_vec, physicsClientId=self.client_id)


    def visualize_angles(self, inv_T, upper, lower):
        og = np.matmul(inv_T, np.array((0, 0, 1)))
        og = og[:2] / og[2]
        og_x = np.matmul(inv_T, np.array((upper[0], upper[1], 1)))
        og_x = og_x[:2] / og_x[2]
        og_y = np.matmul(inv_T, np.array((lower[0], lower[1], 1)))
        og_y = og_y[:2] / og_y[2]

        if self.x_angle < 0:
            self.x_angle = self._p.addUserDebugLine([og[0], og[1], 0.02], [og_x[0], og_x[1], 0.02], [0, 0, 1],
                                                  physicsClientId=self.client_id)
            self.y_angle = self._p.addUserDebugLine([og[0], og[1], 0.02], [og_y[0], og_y[1], 0.02], [0, 0, 1],
                                                  physicsClientId=self.client_id)
        else:
            self._p.addUserDebugLine([og[0], og[1], 0.02], [og_x[0], og_x[1], 0.02], [0, 0, 1],
                                     replaceItemUniqueId=self.x_angle, physicsClientId=self.client_id)
            self._p.addUserDebugLine([og[0], og[1], 0.02], [og_y[0], og_y[1], 0.02], [0, 0, 1],
                                     replaceItemUniqueId=self.y_angle, physicsClientId=self.client_id)

    def visualize_vector(self, inv_T, vec):
        og = np.matmul(inv_T, np.array((0, 0, 1)))
        og = og[:2] / og[2]
        og_x = np.matmul(inv_T, np.array((vec[0], vec[1], 1)))
        og_x = og_x[:2] / og_x[2]

        if self.x_angle < 0:
            self.single_vec = self._p.addUserDebugLine([og[0], og[1], 0.02], [og_x[0], og_x[1], 0.02], [0, 0, 1],
                                                  physicsClientId=self.client_id)
        else:
            self._p.addUserDebugLine([og[0], og[1], 0.02], [og_x[0], og_x[1], 0.02], [0, 0, 1],
                                     replaceItemUniqueId=self.single_vec, physicsClientId=self.client_id)

    def debug_gui_target(self, t_pose):
        # r = R.from_euler('xyz',ori, degrees=False)
        # t_pose = np.matmul(pose, r.as_matrix())
        if self.x_id < 0:
            self.x_id = self._p.addUserDebugLine(t_pose, [t_pose[0] + 0.1, t_pose[1], t_pose[2]], [1, 0, 0],
                                                 physicsClientId=self.client_id)
            self.y_id = self._p.addUserDebugLine(t_pose, [t_pose[0], t_pose[1] + 0.1, t_pose[2]], [0, 1, 0],
                                                 physicsClientId=self.client_id)
            self.z_id = self._p.addUserDebugLine(t_pose, [t_pose[0], t_pose[1], t_pose[2] + 0.1], [0, 0, 1],
                                                 physicsClientId=self.client_id)
        else:
            self._p.addUserDebugLine(t_pose, [t_pose[0] + 0.1, t_pose[1], t_pose[2]], [1, 0, 0],
                                     replaceItemUniqueId=self.x_id,
                                     physicsClientId=self.client_id)
            self._p.addUserDebugLine(t_pose, [t_pose[0], t_pose[1] + 0.1, t_pose[2]], [0, 1, 0],
                                     replaceItemUniqueId=self.y_id,
                                     physicsClientId=self.client_id)
            self._p.addUserDebugLine(t_pose, [t_pose[0], t_pose[1], t_pose[2] + 0.1], [0, 0, 1],
                                     replaceItemUniqueId=self.z_id,
                                     physicsClientId=self.client_id)
    def show_debug_workspace(self, ws):
        self._p.addUserDebugLine([ws[0][0], ws[1][0], ws[2][0]], [ws[0][1], ws[1][0], ws[2][0]], [1, 0, 0],physicsClientId=self.client_id)
        self._p.addUserDebugLine([ws[0][1], ws[1][0], ws[2][0]], [ws[0][1], ws[1][1], ws[2][0]], [1, 0, 0],physicsClientId=self.client_id)
        self._p.addUserDebugLine([ws[0][1], ws[1][1], ws[2][0]], [ws[0][0], ws[1][1], ws[2][0]], [1, 0, 0],physicsClientId=self.client_id)
        self._p.addUserDebugLine([ws[0][0], ws[1][1], ws[2][0]], [ws[0][0], ws[1][0], ws[2][0]], [1, 0, 0],physicsClientId=self.client_id)
