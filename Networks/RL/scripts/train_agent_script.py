from stable_baselines3 import TD3, PPO, SAC
from sb3_contrib import TQC
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.noise import NormalActionNoise
from custom_callbacks import SavingCallback, ProgressBarManager, CurriculumCallback
import os
import gym, push_gym
import numpy as np
import torch
import tensorflow as tf
from stable_baselines3.common.env_util import make_vec_env
from policy_network import CNN



def load_encoder(model_path):
    model = tf.keras.models.load_model(model_path)
    print("done with loading of model ", model_path)
    return model

def get_callbacks(eval_env, log_dir, file_path):
    eval_callback = EvalCallback(eval_env, best_model_save_path=log_dir, log_path=log_dir, eval_freq=30000, n_eval_episodes=15)
    save_callback = SavingCallback(log_dir, file_path[0], file_path[1], file_path[2], curric_file=file_path[3], curric_task_file=file_path[4], save_freq=50000)
    test_callback = CurriculumCallback(eval_env)
    return [eval_callback, save_callback, test_callback]
    return [eval_callback, save_callback]

def get_model(algo, env, policy):
    action_noise = NormalActionNoise(mean=np.zeros(env.action_space.shape[-1]), sigma=0.4 * np.ones(env.action_space.shape[-1]))
    if algo == "TQC":
        policy_kwargs = dict(n_critics=2, activation_fn=torch.nn.ReLU, net_arch=[512, 256, 128])
        return TQC(policy, env, learning_rate=1e-4, tensorboard_log="../tensorboard_logs/", batch_size=512, tau=0.005,
                   gamma=0.98, buffer_size=int(1e6), train_freq=(10, 'step'), learning_starts=10000,
                   top_quantiles_to_drop_per_net=2, gradient_steps=-1,  policy_kwargs=policy_kwargs,
                   action_noise=action_noise, verbose=1, device="cuda")
    elif algo == "TD3":
        policy_kwargs = dict(n_critics=3, activation_fn=torch.nn.ReLU, net_arch=[512, 256, 128])
        return TD3(policy, env, learning_rate=1e-3, tensorboard_log="../tensorboard_logs/", batch_size=256,
                   tau=0.001, gamma=0.95, action_noise=action_noise, buffer_size=int(1e6), train_freq=(5, 'step'),
                   learning_starts=10000, policy_kwargs=policy_kwargs, device="cuda")
    elif algo == "SAC":
        return SAC(policy, env, learning_rate=1e-3, tensorboard_log="../tensorboard_logs/", batch_size=256,
                   tau=0.001, gamma=0.95, action_noise=action_noise, buffer_size=int(1e6), train_freq=(5, 'step'),
                   learning_starts=10000, gradient_steps=-1)
    elif algo == "PPO":
        return PPO(policy, env, learning_rate=1e-5, tensorboard_log="../tensorboard_logs/", batch_size=256,
                   gamma=0.95, gae_lambda=0.9, use_sde=True, vf_coef=0.5, max_grad_norm=0.5,
                   sde_sample_freq=4, ent_coef=0.0, n_epochs=20, n_steps=750)



def get_algorithm(algo):
    if algo == "TQC": return TQC
    elif algo == "TD3": return TD3
    elif algo == "SAC": return SAC
    elif algo == "PPO": return PPO

def train_model(env_file_name, train_file_name, obj_creation_file_name, curric_task_file_name, tf_model_path,
                log_dir_name, tensorboard_log_name, load_model, obstacle_num, env_name, max_timesteps, curriculum,
                algorithm, arm_position, policy):
    for gpu in tf.config.experimental.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)
    #torch.device('cpu')
    torch.set_num_threads(2)

    os.makedirs(log_dir_name, exist_ok=True)
    file_path = [env_file_name, train_file_name, obj_creation_file_name, curric_task_file_name]
    saved_model_path = log_dir_name + "intermediate_saved_model"

    tf_model = load_encoder(tf_model_path)

    # Initialize environments
    env = gym.make(env_name)
    env = make_vec_env(env, n_envs=1)
    env.envs[0].env.setup_for_RL(tf_model, obstacle_num, arm_position, curriculum, with_evaluation=False)
    eval_env = gym.make('pushing-v0')
    eval_env = make_vec_env(eval_env, n_envs=1)
    eval_env.envs[0].env.setup_for_RL(tf_model, obstacle_num, arm_position, curriculum, with_evaluation=False,
                                      is_eval_env=False)

    if load_model:
        algo = get_algorithm(algorithm)
        model = algo.load(saved_model_path).set_env(env)
    else:
        model = get_model(algorithm, env, policy, False)
    try:
        with ProgressBarManager(max_timesteps) as prog_cb:
            model.learn(total_timesteps=max_timesteps, tb_log_name=tensorboard_log_name,
                        callback=get_callbacks(eval_env, log_dir_name, file_path, curriculum) + [prog_cb],
                        reset_num_timesteps=False)
    except KeyboardInterrupt:
        pass

    model.save(log_dir_name + "final_model")
    env._p.unloadPlugin(env.plugin)
    eval_env._p.unloadPlugin(eval_env.plugin)
    env.close()
    eval_env.close()
