import numpy as np
import h5py
import matplotlib.pyplot as plt

def preprocess_images(image):
    unify = image/255#tf.image.convert_image_dtype(image, tf.float32)
    current_min, current_max = np.amin(unify), np.amax(unify)
    normed_min, normed_max = 0, 1
    x_normed = (unify - current_min) / (current_max - current_min)
    x_normed = x_normed * (normed_max - normed_min) + normed_min
    return x_normed

h5_save_path = "/home/user/dengler/Documents/robot_arm_RL/pushing_net/push_gym/push_gym/depth_data/depth_data_300000_3_256.h5"
print("Load np dataset")
h5f = h5py.File(h5_save_path, "r")
print("done")

dset = h5f["data_" + str(1)]
np_data = np.asarray(dset[:])
processed = preprocess_images(np_data)
plt.imshow(processed)
plt.show()
#for i in  range(len(h5f)):
#    dset = h5f["data"+str(i)]
#    print(dset)
#    print(h5f[i])
#data = h5f["dataset"][:]
#print(data)
