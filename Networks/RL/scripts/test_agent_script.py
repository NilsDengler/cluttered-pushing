from stable_baselines3 import TD3, PPO, SAC
from sb3_contrib import TQC
import gym, push_gym
import tensorflow as tf
from tqdm import tqdm
from stable_baselines3.common.env_util import make_vec_env
import json
import os

for gpu in tf.config.experimental.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(gpu, True)

def load_encoder(model_path):
    model = tf.keras.models.load_model(model_path)
    print("done with loading of model ", model_path)
    return model

def test_agent(tf_model_path, evaluation_save_path, log_dir_name, obstacle_num, env_name, arm_position, test_baseline,
               json_file_path):
    
    # check log dir exists
    if not os.path.exists(log_dir_name):
        raise Exception(f'log dir {log_dir_name} does not exist')
    
    # check tf model exists
    if not os.path.exists(tf_model_path):
        raise Exception(f'tf model {tf_model_path} does not exist')

    #load VAE encoder
    tf_model = load_encoder(tf_model_path)
    #specify evaluation file
    evaluation_samples = None
    with_json = False
    if with_json:
        # check json file exists
        if not os.path.exists(json_file_path):
            raise Exception(f'json file {json_file_path} does not exist')
        with open(os.path.join(os.path.dirname(__file__), json_file_path)) as json_file:
            evaluation_samples = json.load(json_file)

    env = make_vec_env(env_name, n_envs=1)
    env.envs[0].env.setup_for_RL(tf_model, obstacle_num, arm_position, False, evaluation_save_path, False)
    eval_max_range = 1000
    if evaluation_samples:
        eval_max_range = len(evaluation_samples['samples'])
    if test_baseline:
        for i in tqdm(range(eval_max_range)):
            if evaluation_samples:
                env.envs[0].env.evaluation_sample = evaluation_samples['samples'][i]
            _ = env.reset()
            _ = env.envs[0].env.baseline.create_corridors()

    else:
        if os.path.exists(log_dir_name + "intermediate_saved_model.zip"):
            saved_model_path = log_dir_name + "intermediate_saved_model"
        elif os.path.exists(log_dir_name + "final_model.zip"):
            print("\033[93m" + f'intermediate_saved_model does not exist in {log_dir_name}' + "\033[0m")
            print("\033[93m" + f'Trying final_model' + "\033[0m")
            saved_model_path = log_dir_name + "final_model"
        else:
            raise Exception(f'No intermediate or final model found in {log_dir_name}')

        model = TQC.load(saved_model_path, env=env)
        for i in tqdm(range(eval_max_range)):
            if evaluation_samples:
                env.envs[0].env.evaluation_sample = evaluation_samples['samples'][i]
            obs = env.reset()
            if i == eval_max_range - 1:
                env.envs[0].env.save_evaluations = True
            Done = False
            while not Done:
                action, _states = model.predict(obs, deterministic=True)
                obs, reward, Done, info = env.step(action)
    env.envs[0].env.close()
