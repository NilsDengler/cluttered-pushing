Train: True
data_file_path: "/home/user/dengler/Documents/robot_arm_RL/pushing_net/push_gym/push_gym/depth_data/local_window_data_243991_64_from_256_3_without_arm_and_obj_fliped_with_zeros.h5" #depth_data_703125_3_256.h5"
model_save_path: "../saved_model/model_infcuda3_larger_local_window_wo_arm_and_obj_encoder_64_from_256_with_zeros_sample_new_try/"
checkpoint_path: "../saved_model/checkpoint_infcuda3_larger_local_window_wo_arm_and_obj_encoder_64_from_256_with_zeros_sample_new_try/"
eval_picture_save_path: "../eval_imgs_infcuda3_larger_local_window_wo_arm_and_obj_encoder_64_from_256_with_zeros_sample_new_try/"
parameter_save_path: "parameters.yaml"
batch_size: 256
learning_rate: 1e-4
latent_dim: 32
kl_weight: 0.5
model_save_rate: 1
picture_save_rate: 1
indipendent: "bernoulli" # normal, bernoulli
epochs: 1000
img_size: 64
train_3d: False
train_astar: False #True
depth_data_file_path: "/home/user/dengler/Documents/robot_arm_RL/pushing_net/push_gym/push_gym/astar_data/depth_data_237705.h5"
heat_data_file_path: "/home/user/dengler/Documents/robot_arm_RL/pushing_net/push_gym/push_gym/astar_data/astar_data_237705.h5"
coord_data_file_path: "/home/user/dengler/Documents/robot_arm_RL/pushing_net/push_gym/push_gym/astar_data/coord_data_237705.h5"
encoder_save_path: "/home/user/dengler/Documents/robot_arm_RL/pushing_net/Networks/VAE/saved_model/checkpoint_infcuda2_bernoulli_32lat_05weight_new_data/"

