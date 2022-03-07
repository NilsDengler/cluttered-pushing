import numpy as np
import h5py

def h5_to_np(h5_data, np_data):
    for i in range(len(h5_data)):
        dset = h5_data["data_" + str(i)]
        np_data[i] = np.asarray(dset[:])
    return np_data

def np_to_h5(h5_save_path, np_data):
    h5_file = h5py.File(h5_save_path, "w")
    for i, d in enumerate(np_data):
        h5_file.create_dataset("data_" + str(i), data=d)
    h5_file.close()
    return

h5_load_path_1 = "/home/user/dengler/Documents/robot_arm_RL/pushing_net/push_gym/push_gym/astar_data/coord_data_118877.h5"
h5_load_path_2 = "/home/user/dengler/Documents/robot_arm_RL/pushing_net/push_gym/push_gym/astar_data/coord_data_118828.h5"
img_size = 256
print("load h5")
h5f_1 = h5py.File(h5_load_path_1, "r")
h5f_2 = h5py.File(h5_load_path_2, "r")
print("create np")
np_data_1 = np.zeros((len(h5f_1), 4), dtype=np.uint8)
np_data_2 = np.zeros((len(h5f_2), 4), dtype=np.uint8)
print("transfer h5 to np")
np_data_1 = h5_to_np(h5f_1, np_data_1)
np_data_2 = h5_to_np(h5f_2, np_data_2)

h5f_1.close()
h5f_2.close()
print("concat np")
np_data_final = np.concatenate((np_data_1, np_data_2), axis=0)
np_data_1 = "Free"
np_data_2 = "Free"
print(np_data_final.shape)
h5_save_path = "/home/user/dengler/Documents/robot_arm_RL/pushing_net/push_gym/push_gym/astar_data/coord  _data_" + \
               str(np_data_final.shape[0])+".h5"
np_to_h5(h5_save_path, np_data_final)
np_data_final= "Free"
