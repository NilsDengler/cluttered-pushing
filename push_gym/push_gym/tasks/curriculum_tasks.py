import numpy as np

class Curriculum:
    def __init__(self, sim, utils, wsb=[-np.inf, np.inf], p_id=0):
        self._p = p_id
        self._workspace_bounds = wsb
        self.sim = sim
        self.utils = utils

    def close_to_far_curriculum(self, current_curriculum_step):
        distances = [0.06, 0.08, 0.13, 0.18, 0.23, 0.28, 0.33, 0.38, 0.43, 0.48, 0.53, 0.60]
        array_length = len(distances)
        prior_step = current_curriculum_step-2
        if prior_step < 0:
            prior_step = 0
        if current_curriculum_step < array_length - 1:
            self.sim.goal_obj_conf = self.utils.generate_random_goal(self.sim.pushing_object_id, 2,
                                                                     distances[current_curriculum_step],
                                                                     distances[prior_step])
        else: self.sim.goal_obj_conf = self.utils.generate_random_goal(self.sim.pushing_object_id, 2, 0.6, 0.2)