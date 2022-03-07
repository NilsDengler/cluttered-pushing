import numpy as np
import h5py

np_data_path = "/home/user/dengler/Documents/robot_arm_RL/pushing_net/push_gym/push_gym/depth_data/depth_data_302500_3_256.npy"
h5_save_path = "/home/user/dengler/Documents/robot_arm_RL/pushing_net/push_gym/push_gym/depth_data/depth_data_302500_3_256.h5"
print("load dataset")
np_dataset = np.load(np_data_path)
print("done")
print("creating hr5 file")
h5_file = h5py.File(h5_save_path, "w")
for i, d in enumerate(np_dataset):
    h5_file.create_dataset("data_"+str(i), data=d)
print("done")
np_dataset = "Free"
h5_file.close()