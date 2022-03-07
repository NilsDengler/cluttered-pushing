from tqdm.auto import tqdm
import shutil
import os
from stable_baselines3.common.callbacks import BaseCallback

class SavingCallback(BaseCallback):
    def __init__(self, log_dir, model_file, train_file, builder_file, curric_task_file=None, save_freq=100000, verbose=0):
        super(SavingCallback, self).__init__(verbose)
        self.log_dir = log_dir
        self.model_file = model_file
        self.train_file = train_file
        self.builder_file = builder_file
        self.save_freq = save_freq
        self.curric_task_file = curric_task_file

    def _on_training_start(self):
        shutil.copyfile(self.model_file, self.log_dir + "env_file.py")
        shutil.copyfile(self.builder_file, self.log_dir + "builder_file.py")
        shutil.copyfile(self.train_file, self.log_dir + "train_file.py")
        shutil.copyfile(self.curric_task_file, self.log_dir + "curric_task_file.py")

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            print("Save intermediate model and replay buffer")
            self.model.save(os.path.join(self.log_dir, 'intermediate_saved_model'))
            self.model.save_replay_buffer(os.path.join(self.log_dir, "replay_buffer"))

        return True


class ProgressBarCallback(BaseCallback):
    """
    :param pbar: (tqdm.pbar) Progress bar object
    """
    def __init__(self, pbar):
        super(ProgressBarCallback, self).__init__()
        self._pbar = pbar

    def _on_step(self):
        # Update the progress bar:
        self._pbar.n = self.num_timesteps
        self._pbar.update(0)


# this callback uses the 'with' block, allowing for correct initialisation and destruction
class ProgressBarManager(object):
    def __init__(self, total_timesteps):  # init object with total timesteps
        self.pbar = None
        self.total_timesteps = total_timesteps

    def __enter__(self):  # create the progress bar and callback, return the callback
        self.pbar = tqdm(total=self.total_timesteps)

        return ProgressBarCallback(self.pbar)

    def __exit__(self, exc_type, exc_val, exc_tb):  # close the callback
        self.pbar.n = self.total_timesteps
        self.pbar.update(0)
        self.pbar.close()

class CurriculumCallback(BaseCallback):

    def __init__(self, eval_env, verbose=1):
        super(CurriculumCallback, self).__init__(verbose)
        self.current_curriculum_step = 0
        self.curriculum_iteration_steps = self.get_curriculum_iteration_steps()
        self.evaluation_env = eval_env

    def get_curriculum_iteration_steps(self):
        return [100000, 250000, 500000, 900000, 1500000, 2000000, 2400000, 2900000]

    def _on_training_start(self):
        self.training_env.envs[0].env.max_curriculum_iteration_steps = len(self.curriculum_iteration_steps)

    def _on_step(self) -> bool:
        self.evaluation_env.envs[0].env.curriculum_difficulty = self.training_env.envs[0].env.curriculum_difficulty
        if self.current_curriculum_step > len(self.curriculum_iteration_steps)-1:
            self.current_curriculum_step = len(self.curriculum_iteration_steps)-1
        if self.n_calls == self.curriculum_iteration_steps[self.current_curriculum_step]:
            self.training_env.envs[0].env.change_curriculum_difficulty = True
            self.current_curriculum_step += 1

            print("CHANGED CURRICULUM FROM ", str(self.current_curriculum_step-1), " TO ", str(self.current_curriculum_step))
        return True

