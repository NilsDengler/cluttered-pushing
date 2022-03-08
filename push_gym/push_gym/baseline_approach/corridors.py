"""
Code is based on the approach by Krivic et al.
Please refer to them for more information:
https://link.springer.com/article/10.1007/s10514-018-9804-8
"""
#!/usr/bin/python3
import numpy as np
from custom_utils import get_config, euler_from_quat, get_point
import LazyThetaStarPython
import cv2
from skimage.draw import line as skline
from skimage.draw import circle_perimeter as skcircle
import pyastar2d


class Corridors:
    """Implementation of Corridors-Paper."""

    def __init__(self, sim):
        #approach parameter
        self.workspace_size = [256, 256]
        self.object_diameter = 0.06
        self.arm_diameter = self.object_diameter
        self.params = None
        self.target_reached_thres = 0.05
        # static simulation parameters
        self.sim = sim
        self._p = self.sim._p
        self.client_id = self.sim.client_id
        self.pushing_object_id = self.sim.pushing_object_id
        self.robot_arm_id = self.sim.robot_arm_id
        self._robot_tool_center = self.sim._robot_tool_center
        self.initial_path_img = None
        self.with_astar = True
        self.utils = self.sim.utils


    def create_corridors(self):
        """ returns grid after steering robot arm according to paper.
        grid representation:
            0: empty cell
            1: object position
            2: goal position
            3: path
            4: corridor
            5: obstacle
            6: pushing border (space for arm not to move in)
            7: object border (space for object not to move in)
        """
        self.init_obj_conf = get_config(self.pushing_object_id, self._p, self.client_id)
        #self.sim.debug_gui_target(np.asarray(self.sim.goal_obj_conf[0]), np.asarray(euler_from_quat(self.sim.goal_obj_conf[1])))
        borders = self.make_grid()
        if self.with_astar:
            grid_py = np.array(self.gridToPy(), dtype=np.float32)
            path = np.array(pyastar2d.astar_path(grid_py[10:-10], self.get_pos_on_grid(self.init_obj_conf[0]) - np.array([10, 0]),
                                                 self.get_pos_on_grid(self.sim.goal_obj_conf[0])- np.array([10, 0])))  # , allow_diagonal=True)
            path[:, 0] += 10
            self.grid[path[:, 0], path[:, 1]] = 3
        else:
            grid_cpp = self.gridToCPP()
            path, length = LazyThetaStarPython.FindPath(np.flip(self.get_pos_on_grid(self.init_obj_conf[0])).tolist(),
                                                        np.flip(self.get_pos_on_grid(self.sim.goal_obj_conf[0])).tolist(),
                                                        grid_cpp.flatten().tolist(), int(self.workspace_size[1]),
                                                        int(self.workspace_size[0]))
            path = np.flip(np.reshape(np.array(path), (-1, 2)), axis=1)
            #use for theta*
            for idx in range(0, len(path) - 1):
                self.grid = cv2.line(self.grid, (path[idx,1], path[idx,0]), (path[idx+1,1], path[idx+1,0]), 3,thickness=2)
        # visualize corridor
        #self.make_corridor(path)

        initial_shortest_path = path
        self.initial_path_img = self.grid_to_pic()

        target = self.look_ahead_target(self.get_pos_on_grid( self.init_obj_conf[0]), path, borders, starting=True)
        if self.sim.with_evaluation == False:
            x_trans, y_trans = np.random.uniform(0.08, 0.1) * np.random.choice([-1, 1]), np.random.uniform(0.08, 0.1) * np.random.choice([-1, 1])
            arm_pose = np.asarray(get_point(self.pushing_object_id, self._p, self.client_id)) - [x_trans, y_trans, 0.01]
            target_joint_states = self.sim.get_ik_joints(arm_pose, [np.pi, 0, 0], self._robot_tool_center)[:6]
            self.sim._reset_arm(target_joint_states)

        self.alphas = []
        self.error_trace = []
        self.A_k = []
        self.likelihood = 0
        self.sec_likelihood = 0

        self.arm_pose = np.array(self.sim._get_link_world_pose(self.robot_arm_id, self._robot_tool_center)[0][:2])
        self.prior = self.get_prior()
        prior_conf = np.array(get_config(self.pushing_object_id, self._p, self.client_id)[0][:2])
        self.moved = False
        reached = False
        index = 0
        successes = 0
        max_steps = 1200
        while (not reached) and (index < max_steps):
            self.arm_pose = np.array(self.sim._get_link_world_pose(self.robot_arm_id, self._robot_tool_center)[0][:2])
            self.old_arm_pose = np.array(self.sim._get_link_world_pose(self.robot_arm_id, self._robot_tool_center)[0])
            self.prior_target = target
            target = self.look_ahead_target(self.get_pos_on_grid(prior_conf), path, borders)
            self.alphas.append(self.abs_angle(self.angle(self.get_real_pos(target) - prior_conf, prior_conf - self.arm_pose)))
            self.pushing(prior_conf)
            #contact is true when arm touches object, otherwise false
            collision = self.sim.utils.check_for_collision_object()
            collision = (collision or self.sim.utils.check_for_collision_EE())
            obj_contact = self.sim.utils.object_contact()
            if obj_contact and not self.sim.first_object_contact:
                self.sim.first_object_contact = True
            self.new_arm_pose = np.array(self.sim._get_link_world_pose(self.robot_arm_id, self._robot_tool_center)[0])
            current_conf = np.array(get_config(self.pushing_object_id, self._p, self.client_id)[0][:2])
            if np.linalg.norm(current_conf - prior_conf) > 0.0001:
                self.moved = True
                self.error_trace.append(self.abs_angle(self.angle(self.get_real_pos(self.prior_target) - prior_conf, current_conf - prior_conf)))
                if (len(self.error_trace) > 1) and (min(abs(0 - self.error_trace[-1]), abs(2*np.pi - self.error_trace[-1])) < min(abs(0 - self.error_trace[-2]), abs(2*np.pi - self.error_trace[-2]))):
                    successes += 1
                    self.likelihood = (successes - 1)*self.likelihood/successes + 1/successes*np.exp(self.params[:, :, 1]*np.cos(self.alphas[-1] - self.params[:, :, 0]))/(2*np.pi*np.i0(self.params[:, :, 1]))
                    self.A_k.append(self.alphas[-1])
            if self.distance(current_conf, self.sim.goal_obj_conf[0][:2]) < self.target_reached_thres:
                reached = True
            index += 1
            if self.sim.with_evaluation:
                self.sim.eval.collect_evaluation_data(current_conf, prior_conf, self.new_arm_pose, self.old_arm_pose,
                                                      obj_contact, collision)
            prior_conf = current_conf

        self.sim.eval.calculate_evaluation_data_baseline(reached, current_conf, initial_shortest_path)
        self.sim.eval.write_evaluation(is_RL=False)
        return self.grid

    def pushing(self, object):
        K_mu = 0.05 #0.01
        K_gamma = 0.1 #0.1
        act_push, act_relocate, alpha_p = self.act_push()


        i_vector = (object - self.arm_pose)/np.linalg.norm(object - self.arm_pose)
        d_push = i_vector

        j_angle = -np.pi/2 + np.arctan2(i_vector[1], i_vector[0])
        j_vector = np.array([np.cos(j_angle), np.sin(j_angle)])
        d_relocate = j_vector

        if np.sin(alpha_p - self.alphas[-1]) < 0:
            d_relocate *= -1

        if (act_relocate > 0.6) and (np.linalg.norm(object - self.arm_pose) > 0.07) and (not self.moved):
            v_ref = 1.0*d_push + act_relocate*d_relocate
        elif (act_relocate > 0.6) and (np.linalg.norm(object - self.arm_pose) < 0.07):
            v_ref = -0.1*act_relocate*d_push + act_relocate*d_relocate
        else:
            v_ref = act_push*d_push + act_relocate*d_relocate

        theta_v = np.arctan2(v_ref[1], v_ref[0])
        if len(self.error_trace) == 0:
            theta_u = theta_v
        else:
            theta_u = theta_v - K_mu*sum(self.error_trace)/len(self.error_trace) - K_gamma*self.error_trace[-1]
        action = np.array([np.cos(theta_u), np.sin(theta_u), 0])
        self.sim._move_to_position(action)
        self.sim.step_simulation(1)

    def act_push(self):
        if len(self.A_k) > 0:
            p_Ak = len(self.A_k)/(len(self.alphas) - 1)
            posterior = self.likelihood*self.prior/p_Ak
            mu, k = self.params[int(np.argmax(posterior)/101), np.argmax(posterior)%101]
            probs = np.exp(k*np.cos(self.A_k - mu))/(2*np.pi*np.i0(k))
            alpha_p = np.average(self.A_k, weights=probs)
        else:
            posterior = self.prior
            mu, k = self.params[int(np.argmax(posterior)/101), np.argmax(posterior)%101]
            alpha_p = 0


        if (min(abs(0 - (alpha_p - self.alphas[-1])), abs(2*np.pi - abs(alpha_p - self.alphas[-1]))) < np.pi/8) and (min(abs(0 - (alpha_p - self.alphas[-1])), abs(2*np.pi - abs(alpha_p - self.alphas[-1]))) < min(abs(0 - self.alphas[-1]), abs(2*np.pi - abs(self.alphas[-1])))): # (abs(self.abs_angle(alpha_p)) < np.pi/2) and  #(min(abs(0 - alpha_p), abs(2*np.pi - abs(alpha_p))) < np.pi/8) and
            prob = np.exp(k*np.cos(alpha_p - self.alphas[-1] - mu))/(2*np.pi*np.i0(k))
        else:
            prob = np.exp(k*np.cos(self.alphas[-1] - mu))/(2*np.pi*np.i0(k))
        act_push = prob/(np.exp(k*np.cos(mu - mu))/(2*np.pi*np.i0(k)))
        act_relocate = np.sqrt(1 - np.square(act_push))
        return act_push, act_relocate, alpha_p

    def get_prior(self):
        """Calculate prior p(mu, k)"""
        R_0 = 5.84
        c = 7.0
        phi = 0

        K = 0
        k = 0
        quotient = 0
        #integrate bessel function over k until quotient basically infinite
        while True:
            if np.power(np.i0(k), c) < quotient:
                break
            else:
                quotient = np.power(np.i0(k), c)
                K += np.i0(R_0*k)/np.power(np.i0(k), c)
                k += 1

        self.params = np.ones((361, 101, 2)) #mu, k
        self.params[:, :, 0] *= np.transpose(np.array([np.linspace(0, 2*np.pi, 361) for i in range(101)])) #mu
        self.params[:, :, 1] *= np.array([np.linspace(0, 50, 101) for i in range(361)]) #k
        return 1/K*(np.exp(R_0*self.params[:, :, 1]*np.cos(phi - self.params[:, :, 0]))/2*np.pi*np.power(np.i0(self.params[:, :, 1]), c))


    def look_ahead_target(self, start, path, borders, starting=False): #need to add inner borders
        """receives path as input and aims to find the best look ahead target
        by iterating through the path points from furthest to closest and
        finding furthest point reachable without a collision, eg walking through
        zone. This corresponds to the relaxed strategy in the paper.
        """
        start = np.array(start)
        if borders is not None:
            c_index = np.argmin(np.sqrt(np.square(np.array(path[:, 0] - start[0], dtype=np.float32)) + np.square(np.array(path[:, 1] - start[1], dtype=np.float32))))
            c = path[c_index]
            w_index = np.argmin(np.sqrt(np.square(np.array(borders[:, 0] - start[0], dtype=np.float32)) + np.square(np.array(borders[:, 1] - start[1], dtype=np.float32))))
        else:
            return path[-1]

        tight = np.array([-1, -1])
        supertight = np.array([-1, -1])
        obst_idxs = np.argwhere(self.grid==5)

        if not starting:
            arm = self.get_pos_on_grid(self.arm_pose)

        for i, point in enumerate(np.flip(path[max(c_index, 1)::2], axis=0)):
            if starting:
                startline = start - point
                straight_start = np.array(start + startline/np.linalg.norm(startline)*256*self.arm_diameter*1.5, dtype=int)
                if (np.all(straight_start > -1) and np.all(straight_start < 256)) and ((self.grid[straight_start[0], straight_start[1]] == 5)): # or (self.grid[straight_start[0], straight_start[1]] == 6)
                    continue
            ot = point - start #object to target
            line = skline(start[0], start[1], point[0], point[1])
            idxs = np.intersect1d(np.argwhere(line[0][:] < self.workspace_size[0]), np.argwhere(line[1][:] < self.workspace_size[1]))
            line = (line[0][idxs], line[1][idxs])
            if np.any(supertight < 0):
                if np.all(self.grid[line] != 5) and self.d(point, start, obst_idxs):
                    supertight = point
            if np.any(tight < 0):
                if np.all(self.grid[line] != 5) and np.all(self.grid[line] != 6):
                    tight = point
            if np.any(self.grid[line] >= 5):
                continue
            return point

        if np.all(tight > 0) and not starting:
            startline = start - tight
            straight_start = np.array(start + startline/np.linalg.norm(startline)*256*self.arm_diameter*1.5, dtype=int)
            line = skline(straight_start[0], straight_start[1], arm[0], arm[1])
            if (np.all(straight_start > -1) and np.all(straight_start < 256)) and np.all(self.grid[line] != 5):
                return tight

        if np.all(supertight > 0) and not starting:
            obstacles = np.argwhere(self.grid == 5)
            superline = skline(start[0], start[1], supertight[0], supertight[1])
            idxs = np.intersect1d(np.argwhere(superline[0][:] < self.workspace_size[0]), np.argwhere(superline[1][:] < self.workspace_size[1]))
            superline = (superline[0][idxs], superline[1][idxs])
            critical = np.argwhere(self.grid[superline] == 6)
            if len(critical) > 0:
                distances = [np.sqrt((obstacles[:, 0] - superline[0][i])**2 + (obstacles[:, 1] - superline[1][i])**2) for i in critical]
                minDist = np.min(np.array(distances))
            else:
                minDist = 15
            startline = start - supertight
            straight_start = np.array(start + startline/np.linalg.norm(startline)*256*self.arm_diameter*1.5, dtype=int)
            line = skline(straight_start[0], straight_start[1], arm[0], arm[1])
            idxs = np.intersect1d(np.argwhere(line[0][:] < self.workspace_size[0]), np.argwhere(line[1][:] < self.workspace_size[1]))
            line = (line[0][idxs], line[1][idxs])
            if minDist > 7 and np.all(self.grid[line] != 5):
                return supertight

        if (self.grid[c[0], c[1]] < 5) and not starting:
            startline = start - c
            straight_start = np.array(start + startline/np.linalg.norm(startline)*256*self.arm_diameter*1.5, dtype=int)
            line = skline(straight_start[0], straight_start[1], arm[0], arm[1])
            idxs = np.intersect1d(np.argwhere(line[0][:] < self.workspace_size[0]), np.argwhere(line[1][:] < self.workspace_size[1]))
            line = (line[0][idxs], line[1][idxs])
            if np.all(self.grid[line] != 5):
                return path[min(c_index + 5, len(path) - 1)]
        if np.all(tight > 0):
            return tight
        return self.prior_target

    def make_corridor(self, path):
        green = 3
        width = 256*(self.arm_diameter + self.object_diameter)/0.907

        edges = [[], []]
        for i, point in enumerate(path[1:-1]):
            tangent = path[i+2] - path[i]
            tangent = np.array([-tangent[1], tangent[0]])/np.linalg.norm(tangent)
            # out = [point + width*tangent, point - width*tangent]
            edges[0].append(np.array(point + width*tangent, dtype=int))
            edges[1].append(np.array(point - width*tangent, dtype=int))

        points = None
        for site in edges:
            for i in range(len(site) - 1):
                line = skline(site[i][0], site[i][1], site[i+1][0], site[i+1][1])
                idxs = np.intersect1d(np.intersect1d(np.argwhere(line[0][:] > -1), np.argwhere(line[0][:] < self.workspace_size[0])), np.intersect1d(np.argwhere(line[1][:] > -1), np.argwhere(line[1][:] < self.workspace_size[1])))
                line = (line[0][idxs], line[1][idxs])
                # self.grid[line] = 1
                if len(line) == 0:
                    continue
                if points is None:
                    points = np.transpose(np.asarray(line))
                else:
                    points = np.concatenate((points, np.transpose(np.asarray(line))), 0)

        circle = np.transpose(np.asarray(skcircle(path[0, 0], path[0, 1], radius=int(width), shape=self.workspace_size)))
        if points is None:
            points = circle
        else:
            points = np.concatenate((points, circle), 0)
        circle = np.transpose(np.asarray(skcircle(path[-1, 0], path[-1, 1], radius=int(width), shape=self.workspace_size)))
        if points is None:
            points = circle
        else:
            points = np.concatenate((points, circle), 0)

        current = max(points[:, 0])
        while current in points[:, 0]:
            space = points[np.where(points[:, 0] == current), :][0]
            if len(space[:, 0]) == 1:
                if self.grid[current, space[0,1]] == 0:
                    self.grid[current, space[0, 1]] = green
            else:
                empty = np.where(self.grid[current, np.min(space[:, 1]):np.max(space[:, 1])] == 0) + np.min(space[:,1])
                self.grid[current, empty] = green
            current -= 1

    def make_grid(self):
        width, height = self.workspace_size[0] + 80, self.workspace_size[1] + 80
        self.grid = np.zeros((width, height), dtype=np.int8)
        start = np.flip(self.get_pos_on_grid(self.init_obj_conf[0], starting=True))
        self.grid = cv2.rectangle(self.grid, tuple(start + 8), tuple(start - 8), 2, -1)
        end = np.flip(self.get_pos_on_grid(self.sim.goal_obj_conf[0], starting=True))
        self.grid = cv2.rectangle(self.grid, tuple(end + 8), tuple(end - 8), 1, -1)
        borders = None
        for i, obstacle in enumerate(self.sim.obstacles):
            obst_conf = get_config(obstacle, self._p, self.client_id)
            obst_yaw = euler_from_quat(obst_conf[1])
            obst_corners = self.get_corners(obst_conf[0], self.utils.obst_diameters[i], obst_yaw[-1], corridor=True)
            if obst_corners != None:
                border = self.connect_corners(obst_corners, 5, corridor=True)
                if borders is None:
                    borders = border
                else:
                    borders = np.concatenate((borders, border), 0)
        self.grid = self.grid[40:-40, 40:-40]
        return borders

    def get_corners(self, pos, diameter, yaw, corridor=False):
        length = np.sqrt(np.square(self.object_diameter/2) + np.square(diameter/2))
        arm_length = np.sqrt(2*np.square(self.arm_diameter/2))
        object_length = np.sqrt(2*np.square(self.object_diameter/2))
        corners = []
        pushing_corners = []
        corridor_corners = []
        for i in range(4):
            corner = [0, 0]
            pushing_corner = [[0,0], [0,0]]
            corridor_corner = [[0,0], [0,0]]
            if i%2 == 1:
                angle = np.arccos((self.object_diameter/2)/length) + np.deg2rad(90*i) + yaw
                corner[0] = pos[0] + np.cos(angle)*length
                corner[1] = pos[1] + np.sin(angle)*length
                if corridor:
                    pushing_corner[0][0] = corner[0] + np.cos(np.deg2rad(90)*i + yaw)*arm_length
                    pushing_corner[0][1] = corner[1] + np.sin(np.deg2rad(90)*i + yaw)*arm_length
                    pushing_corner[1][0] = corner[0] + np.cos(np.deg2rad(90)*((i+1)%4) + yaw)*arm_length
                    pushing_corner[1][1] = corner[1] + np.sin(np.deg2rad(90)*((i+1)%4) + yaw)*arm_length
                    corridor_corner[0][0] = corner[0] + np.cos(np.deg2rad(90)*i + yaw)*(arm_length + object_length)
                    corridor_corner[0][1] = corner[1] + np.sin(np.deg2rad(90)*i + yaw)*(arm_length + object_length)
                    corridor_corner[1][0] = corner[0] + np.cos(np.deg2rad(90)*((i+1)%4) + yaw)*(arm_length + object_length)
                    corridor_corner[1][1] = corner[1] + np.sin(np.deg2rad(90)*((i+1)%4) + yaw)*(arm_length + object_length)
            else:
                angle = np.arccos((diameter/2)/length) + np.deg2rad(90*i) + yaw
                corner[0] = pos[0] + np.cos(angle)*length
                corner[1] = pos[1] + np.sin(angle)*length
                if corridor:
                    pushing_corner[0][0] = corner[0] + np.cos(np.deg2rad(90)*i + yaw)*arm_length
                    pushing_corner[0][1] = corner[1] + np.sin(np.deg2rad(90)*i + yaw)*arm_length
                    pushing_corner[1][0] = corner[0] + np.cos(np.deg2rad(90)*((i+1)%4) + yaw)*arm_length
                    pushing_corner[1][1] = corner[1] + np.sin(np.deg2rad(90)*((i+1)%4) + yaw)*arm_length
                    corridor_corner[0][0] = corner[0] + np.cos(np.deg2rad(90)*i + yaw)*(arm_length + object_length)
                    corridor_corner[0][1] = corner[1] + np.sin(np.deg2rad(90)*i + yaw)*(arm_length + object_length)
                    corridor_corner[1][0] = corner[0] + np.cos(np.deg2rad(90)*((i+1)%4) + yaw)*(arm_length + object_length)
                    corridor_corner[1][1] = corner[1] + np.sin(np.deg2rad(90)*((i+1)%4) + yaw)*(arm_length + object_length)
            corners.append(corner)
            pushing_corners.append(pushing_corner)
            corridor_corners.append(corridor_corner)
        if corridor:
            return corners, pushing_corners, corridor_corners
        return corners

    def connect_corners(self, corners, type, corridor=False):
        """goal is to return a grid containing accurate object representations
            tricky cases:
            -corner is not within grid => obstacle extends beyond boundaries
                => check if corner is within grid, otherwise start at other point
            - there might not be a closed rectangle within the grid, provided that
                one corner extends beyond

            +++APPROACH+++:
            -use corner furthest in the middle, except corners, every grid line
            should be cut at least twice (corners might as well)
            -maybe even start with the highest point? might either be a corner (=> not necessarily another point on line)
            or a normal point on line(unless on border, must be on line with another point)
                => get max,
                => go lower one index at a time and fill between min and max or boundary(extra case when min = max-1 and when point is a corner)
                => stop when no point with lower index is there anymore
            -to deal with objects extending beyond boundaries, use boundaries as helper line,
            otherwise fill space in between lowest and highest index in certain line (as long as lowest or highest is not directly next to each other)
        """
        # connect object corners
        if corridor:
            corners, zone_corners, corridor_corners = corners
        points = None
        for i in range(len(corners)):
            #connect to next corner
            start_cell = self.get_pos_on_grid(corners[i], starting=True)
            end_cell = self.get_pos_on_grid(corners[(i+1)%len(corners)], starting=True)
            #connect cells using bresenham algo
            line = self.bresenham(start_cell, end_cell, pad=80)
            if len(line) == 0:
                continue
            if points is None:
                points = np.asarray(line)
            else:
                points = np.concatenate((points, np.asarray(line)), 0)
        #paint cells within bounding box - point[0] or point[1]?
        current = max(points[:, 0])
        while current in points[:, 0]:
            space = points[np.where(points[:, 0] == current), :][0]
            if len(space[:, 0]) == 1:
                self.grid[current, space[0, 1]] = type
            else:
                self.grid[current, np.min(space[:, 1]):np.max(space[:, 1])] = type
            current -= 1

        #connect corridor corners
        #connect quarter circles
        points = None
        if corridor:
            width = np.sqrt(2*np.square(self.arm_diameter/2))*256/0.907
            for i in range(len(zone_corners)):
                start_cell = self.get_pos_on_grid(zone_corners[i][0], starting=True)
                end_cell = self.get_pos_on_grid(zone_corners[i][1], starting=True)
                grid_corner = self.get_pos_on_grid(corners[i], starting=True)
                circle = np.transpose(np.asarray(skcircle(grid_corner[0], grid_corner[1], radius=int(width), shape=(np.array(self.workspace_size) + 80))))
                for point in circle:
                    if self.grid[point[0], point[1]] == 0:
                        self.grid[point[0], point[1]] = 6

                next_cell = self.get_pos_on_grid(zone_corners[(i+1)%len(zone_corners)][0], starting=True)
                line = self.bresenham(end_cell, next_cell, pad=80)
                # #paint
                for point in line:
                    if self.grid[point[0], point[1]] == 0:
                        self.grid[point[0], point[1]] = 6

                if len(line) == 0:
                    continue
                if points is None:
                    points = np.array(line).reshape((-1, 2))
                else:
                    points = np.concatenate((points, np.reshape(np.array(line),(-1,2))), 0)
                points = np.concatenate((points, np.reshape(np.array(circle),(-1,2))), 0)
            points = np.array(points, dtype=int)
            current = int(max(points[:, 0]))
            while current in points[:, 0]:
                space = points[np.where(points[:, 0] == current), :][0]
                if len(space[:, 0]) == 1:
                    if self.grid[current, space[0,1]] == 0 or self.grid[current, space[0,1]] == 7:
                        self.grid[current, space[0, 1]] = 6
                else:
                    empty = np.where(self.grid[current, np.min(space[:, 1]):np.max(space[:, 1])] == 0) + np.min(space[:,1])
                    cover = np.where(self.grid[current, np.min(space[:, 1]):np.max(space[:, 1])] == 7) + np.min(space[:,1])
                    self.grid[current, empty] = 6
                    self.grid[current, cover] = 6
                current -= 1
            current = int(max(points[:, 1]))
            while current in points[:, 1]:
                space = points[np.where(points[:, 1] == current), :][0] #1?
                if len(space[:, 1]) == 1:
                    if self.grid[space[0,0], current] == 0 or self.grid[space[0,0], current] == 7:
                        self.grid[space[0,0], current] = 6
                else:
                    empty = np.where(self.grid[np.min(space[:, 0]):np.max(space[:, 0]), current] == 0) + np.min(space[:,0])
                    cover = np.where(self.grid[np.min(space[:, 0]):np.max(space[:, 0]), current] == 7) + np.min(space[:,0])
                    self.grid[empty, current] = 6
                    self.grid[cover, current] = 6
                current -= 1

            width += np.sqrt(2*np.square(self.object_diameter/2))*256/0.907
            points = None
            for i in range(len(corridor_corners)):
                start_cell = self.get_pos_on_grid(corridor_corners[i][0], starting=True)
                end_cell = self.get_pos_on_grid(corridor_corners[i][1], starting=True)
                grid_corner = self.get_pos_on_grid(corners[i], starting=True)
                circle = np.transpose(np.asarray(skcircle(grid_corner[0], grid_corner[1], radius=int(width), shape=(np.array(self.workspace_size) + 80))))
                for point in circle:
                    if self.grid[point[0], point[1]] == 0:
                        self.grid[point[0], point[1]] = 7

                next_cell = self.get_pos_on_grid(corridor_corners[(i+1)%len(corridor_corners)][0], starting=True)
                line = self.bresenham(end_cell, next_cell, pad=80)

                # #paint
                for point in line:
                    if self.grid[point[0], point[1]] == 0:
                        self.grid[point[0], point[1]] = 7

                if len(line) == 0:
                    continue
                if points is None:
                    points = np.array(line).reshape((-1, 2))
                else:
                    points = np.concatenate((points, np.reshape(np.array(line),(-1,2))), 0)
                points = np.concatenate((points, np.reshape(np.array(circle),(-1,2))), 0)
            points = np.asarray(points, dtype=int)
            current = max(points[:, 0])
            while current in points[:, 0]:
                space = np.unique(points[np.where(points[:, 0] == current), :][0], axis=0)
                if len(space[:, 0]) == 1:
                    if self.grid[current, space[0, 1]] == 0:
                        self.grid[current, space[0, 1]] = 7
                elif len(space[:, 0]) > 1:
                    empty = np.where(self.grid[current, np.min(space[:, 1]):np.max(space[:, 1])] == 0) + np.min(space[:,1])
                    self.grid[current, empty] = 7
                current -= 1
            return points
        return

    def midpoint_circle(self, start, end, radius):
        curve = []
        rev_curve = []
        dx, dy = radius, 0
        x = start[0]
        y = start[1]
        while dx != dy:
            curve.append([x, start[1] + dy])
            rev_curve.append([end[0] + dy, end[1] + dx - radius])
            dy += 1
            x = int(np.around(np.square(dx) - 2*dy - 1))
        curve.append(rev_curve.reverse())
        return curve

    def bresenham(self, start, end, pad=0):
        #watch out whether starting point is on grid or nah, if neither is, return []
        if start[0] < 0 or start[1] < 0:
            if end[0] < 0 or end[1] < 0:
                return []
            start, end = end, start

        line = []
        x0, y0 = start[0], start[1]
        x1, y1 = end[0], end[1]

        dx = abs(x1-x0)
        sx = 1
        if x1 < x0:
            sx = -1

        dy = -abs(y1-y0)
        sy = 1
        if y1 < y0:
            sy = -1

        err = dx + dy
        current = [x0, y0]
        while True:
            if current[0] >= 0 and current[1] >= 0 and current[0] < (self.workspace_size[0] + pad) and current[1] < (self.workspace_size[1] + pad):
                line.append([current[0], current[1]])
            else:
                break #?
            if current == [x1, y1]:
                break
            e2 = 2*err
            if e2 > dy:
                err += dy
                current[0] += sx
            if e2 < dx:
                err += dx
                current[1] += sy

        return line

    def get_pos_on_grid(self, coords, starting=False):
        "gets global coords and returns indices on grid"
        pos = np.asarray([0,0], dtype=int)
        if starting:
            # pos[0] = int((coords[0] + 0.65625)*(self.workspace_size[0] + 80)/(1.3125)) #int((coords[0] + 0.45)*255/0.9) #
            pos[0] = int((coords[0] + 0.452 + (0.907*336/256 - 0.907)/2)*(self.workspace_size[0] + 80)/(0.907*336/256)) #int((coords[0] + 0.45)*255/0.9) #
            # pos[1] = int((coords[1] + 1.15625)*(self.workspace_size[1] + 80)/(1.3125)) #int((coords[1] + 0.75)*127/0.45)#
            pos[1] = int((coords[1] + 1.05 + (0.903*336/256 - 0.903)/2)*(self.workspace_size[1] + 80)/(0.903*336/256)) #int((coords[1] + 0.75)*127/0.45)#
        else:
            # pos[0] = int((coords[0] + 0.5)*(self.workspace_size[0] - 1)/1) #int((coords[0] + 0.45)*255/0.9) #
            pos[0] = int((coords[0] + 0.452)*(self.workspace_size[0] - 1)/0.907) #int((coords[0] + 0.45)*255/0.9) #
            # pos[1] = int((coords[1] + 1)*(self.workspace_size[1] - 1)/1) #int((coords[1] + 0.75)*127/0.45)#
            pos[1] = int((coords[1] + 1.05)*(self.workspace_size[1] - 1)/0.903) #int((coords[1] + 0.75)*127/0.45)#
        return pos

    def get_real_pos(self, coords):
        "gets grid coordinates and returns center of grid cell in absolute coords"
        pos = np.asarray([0,0], dtype=float)
        # pos[0] = (1/2)*(1.2/self.workspace_size[0]) + coords[0]/self.workspace_size[0]*1 - 0.5
        pos[0] = coords[0]/(self.workspace_size[0] - 1)*0.907 - 0.452
        # pos[1] = (1/2)*(1/(self.workspace_size[1] - 1)) + coords[1]/(self.workspace_size[1] - 1)*1 - 0.9
        pos[1] = coords[1]/(self.workspace_size[1] - 1)*0.903 - 1.05
        return pos

    def grid_dist(self, a, b):
        return np.sqrt(np.square(a[0]-b[0]) + np.square(a[1]-b[1]))

    def distance(self, a, b):
        return np.sqrt(np.square(a[0]-b[0]) + np.square(a[1]-b[1]))

    def angle(self, a, b):
        """returns angle between a and b in radians, operates in two-dim space"""
        return np.arctan2(b[1], b[0]) - np.arctan2(a[1], a[0])

    def abs_angle(self, theta):
        """returns value of theta in interval (-pi, pi]"""
        if (theta > -np.pi) and (theta <= np.pi):
            return theta
        elif theta <= -np.pi:
            return abs(theta + 2*np.pi)
        elif theta > np.pi:
            return - (theta - np.pi)
        else:
            print("error for theta: ", theta)

    def t(self, p, q, r):
        x = p-q
        return np.dot(r-q, x)/np.dot(x, x)

    def d(self, p, q, r):
        t = self.t(p, q, r)
        t[np.argwhere(t < 0)] = 0
        t[np.argwhere(t > 1)] = 1
        # print(t.shape)
        return np.linalg.norm(np.matmul(np.expand_dims(t, axis=1),np.expand_dims(p-q, axis=0))+q-r)

    def grid_to_pic(self, appendix=""):
        """
        grid representation:
            0: empty cell
            1: object position
            2: goal position
            3: path
            4: corridor
            5: obstacle
            6: pushing border (space for arm not to move in)
            7: object border (space for object not to move in)
        """
        height, width = self.workspace_size
        picture = np.zeros((height, width, 3), dtype=np.uint8) #size change (256, 256
        picture[self.grid == 0] = np.asarray([255,255, 255])
        picture[self.grid == 1] = np.asarray([0,0,255])
        picture[self.grid == 2] = np.asarray([0,255,0])
        picture[self.grid == 3] = np.asarray([255,0,255])
        picture[self.grid == 4] = np.asarray([0,255,0])
        picture[self.grid == 5] = np.asarray([255,0,0])
        picture[self.grid == 6] = np.asarray([255,255,0])
        picture[self.grid == 7] = np.asarray([255,255,0])
        return picture
        #plt.imsave("grid" + appendix + ".png", picture)

    def gridToCPP(self, tight=False, supertight=False):
        """
        grid representation:
            0: empty cell
            1: object position
            2: goal position
            3: path
            4: corridor
            5: obstacle
            6: pushing border (space for arm not to move in)
            7: object border (space for object not to move in)
        """
        cpp_grid = self.grid.copy().astype(np.uint8)
        cpp_grid[cpp_grid == 5] = 255
        cpp_grid[cpp_grid < 5] = 0
        #cpp_grid[cpp_grid == 6] = 5
        if tight:
            cpp_grid[cpp_grid == 7] = 1
            cpp_grid[cpp_grid == 6] = 5
        elif supertight:
            cpp_grid[cpp_grid == 7] = 0
            cpp_grid[cpp_grid == 6] = 0
        else:
            cpp_grid[cpp_grid == 7] = 2
            cpp_grid[cpp_grid == 6] = 5
        return cpp_grid

    def gridToPy(self, tight=False):
        """
        grid representation:
            0: empty cell
            1: object position
            2: goal position
            3: path
            4: corridor
            5: obstacle
            6: pushing border (space for arm not to move in)
            7: object border (space for object not to move in)
        """
        cpp_grid = self.grid.copy()
        cpp_grid = np.where(cpp_grid > 4, cpp_grid, 1)
        cpp_grid = np.where(cpp_grid != 5, cpp_grid, 256*256)
        cpp_grid = np.where(cpp_grid != 6, cpp_grid, 200)
        cpp_grid = np.where(cpp_grid != 7, cpp_grid, 50)
        return cpp_grid
