"""
Code is based on
https://github.com/bulletphysics/bullet3/tree/master/examples/pybullet/gym/pybullet_envs
and
https://github.com/openai/gym/tree/master/gym/envs
"""

import pybullet as p
from pybullet_utils import bullet_client
import gym
from gym.utils import EzPickle
import os,sys
from copy import deepcopy
from abc import ABCMeta, abstractmethod
import pybullet_data
import numpy as np
import pkgutil

class BasePybulletEnv(gym.Env, EzPickle, metaclass=ABCMeta):
    def __init__(self, render=False, shared_memory=False, hz=240, use_egl=False):
        EzPickle.__init__(**locals())
        self._p = None
        self.step_size_fraction = hz
        self._urdfRoot = pybullet_data.getDataPath()
        if use_egl and render:
            raise ValueError('EGL rendering cannot be used with GUI .')

        render_option = p.DIRECT
        if render:
            render_option = p.GUI
            if shared_memory:
                render_option = p.SHARED_MEMORY

        self._p = bullet_client.BulletClient(connection_mode=render_option)
        self._egl_plugin = None
        # if use_egl:
        #     assert sys.platform == 'linux', ('EGL rendering is only supported on ''Linux.')
        #     egl = pkgutil.get_loader('eglRenderer')
        #     if egl:
        #         self._egl_plugin = self._p.loadPlugin(egl.get_filename(),'_eglRendererPlugin')
        #     else:
        #         self._egl_plugin = self._p.loadPlugin('eglRendererPlugin')
        #     print('EGL renderering enabled.')

        self._p.configureDebugVisualizer(self._p.COV_ENABLE_GUI, 0)
        self._p.setPhysicsEngineParameter(enableFileCaching=0)
        self._p.setPhysicsEngineParameter(numSolverIterations=200)
        self._p.setTimeStep(1. / self.step_size_fraction)

        if render:
            self._p.resetDebugVisualizerCamera(
                cameraDistance=5.0,
                cameraYaw=0,
                cameraPitch=-89,
                cameraTargetPosition=[0, 0, 0],
            )
            p.resetDebugVisualizerCamera(2.,0,0, cameraTargetPosition=[0,0,1])

        self.client_id = self._p._client
        self._reset_base_simulation()


    def _reset_base_simulation(self):
        self._p.resetSimulation()
        self._p.setGravity(0, 0, -9.81)

    def seed(self, seed=None):
        self._random = np.random.RandomState(seed)
        return seed

    def _get_obj_world_pose(self, id):
        return self._p.getBasePositionAndOrientation(id)

    def _get_link_world_pose(self, body_id, link_id):
        return self._p.getLinkState(body_id, link_id, physicsClientId=self.client_id)

    def step_simulation(self, num_steps):
        for i in range(int(num_steps)):
            self._p.stepSimulation(physicsClientId=self.client_id)
            self._p.performCollisionDetection()
            if self.with_gripper:
                body_id, main_joint_id = self.robot_arm_id, 0
                gripper_joint_position = self._p.getJointState(self.robot_arm_id, self._gripper_joint_indices[main_joint_id])[0]
                self._p.setJointMotorControlArray(self.robot_arm_id, self._gripper_joint_indices[1:], self._p.POSITION_CONTROL,
                                                  [-gripper_joint_position, gripper_joint_position, gripper_joint_position,
                                                   -gripper_joint_position, gripper_joint_position],
                                                    positionGains=np.ones(5), physicsClientId=self.client_id)

    def close(self):
        if self._egl_plugin is not None:
            p.unloadPlugin(self._egl_plugin)
        self._p.disconnect()

    def copy(self):
        return deepcopy(self)

    def render(self, mode="human"):
        print("ERROR! Please choose GUI version of environment!")
