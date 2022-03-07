import numpy as np
import h5py

h5_save_path = "/home/user/dengler/Documents/robot_arm_RL/pushing_net/push_gym/push_gym/depth_data/depth_data_302500_3_256.h5"
print("Load np dataset")
h5f = h5py.File(h5_save_path, "r")
print("done")
print("reading data")
print(h5f.keys())
#for key, value in h5f.items():
#    print(key)
for i in  range(len(h5f)):
    dset = h5f["data"+str(i)]
    print(dset)
#    print(h5f[i])
#data = h5f["dataset"][:]
#print(data)
