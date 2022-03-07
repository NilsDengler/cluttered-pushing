import numpy as np
import matplotlib.pyplot as plt
import cv2

def softmax(X):
    expo = np.exp(X)
    expo_sum = np.sum(np.exp(X))
    return expo/expo_sum

def normalize_fixed(x):
    current_min, current_max = np.amin(x), np.amax(x)#tf.expand_dims(current_range[:, 0], 1), tf.expand_dims(current_range[:, 1], 1)
    print("argmin", current_min, current_max)
    normed_min, normed_max = 0 , 1#tf.expand_dims(normed_range[:, 0], 1), tf.expand_dims(normed_range[:, 1], 1)
    x_normed = (x - current_min) / (current_max - current_min)
    x_normed = x_normed * (normed_max - normed_min) + normed_min
    print(x_normed[x_normed<0.99])
    return x_normed


rgb_data = np.load("/home/user/dengler/Documents/robot_arm_RL/pushing_net/push_gym/push_gym/depth_wo_arm.npy")
d_data = np.load("/home/user/dengler/Documents/robot_arm_RL/pushing_net/push_gym/push_gym/rrt_data/depth_data_10000.npy")
#d_data = np.load("/home/user/dengler/Documents/robot_arm_RL/pushing_net/push_gym/push_gym/depth_data/depth_data_302500_3_256.npy")
h_data = np.load("/home/user/dengler/Documents/robot_arm_RL/pushing_net/push_gym/push_gym/rrt_data/heat_data_10000.npy")
l_data = np.load("/home/user/dengler/Documents/robot_arm_RL/pushing_net/push_gym/push_gym/rrt_data/label_data_10000.npy")
#np.save("/home/user/dengler/Documents/robot_arm_RL/pushing_net/Networks/VAE_Mnist/Network_test_image_256.npy", d_data[0])
print("START ZERO CHECK")
id_deletion_depth_list = []
id_deletion_heat_list = []
#for i in range(len(d_data)):
#    if np.all((d_data[i] == 0)):
#        id_deletion_depth_list.append(i)
#    if np.all((h_data[i][:, :, 0] == 0)):
#        id_deletion_heat_list.append(i)
print(len(id_deletion_depth_list), "DELETION DEPTH IDs")
print(len(id_deletion_heat_list), "DELETION HEAT IDs")
print("END ZERO CHECK")
print("start to delete")
#d_data = np.delete(d_data, id_deletion_heat_list, 0)
#h_data = np.delete(h_data, id_deletion_heat_list, 0)
#l_data = np.delete(l_data, id_deletion_heat_list, 0)
print("done with deletion")
#np.save("/home/user/dengler/Documents/robot_arm_RL/pushing_net/push_gym/push_gym/rrt_data/depth_data_10000_cleaned.npy", d_data)
#np.save("/home/user/dengler/Documents/robot_arm_RL/pushing_net/push_gym/push_gym/rrt_data/heat_data_10000_cleaned.npy", h_data)
#np.save("/home/user/dengler/Documents/robot_arm_RL/pushing_net/push_gym/push_gym/rrt_data/label_data_10000_cleaned.npy", l_data)
# print("start to delete")
# print(len(h_data))
# h_data = np.delete(h_data, id_deletion_list, 0)
# print(len(h_data))
# print("done with deletion")
img_num = 2000
print(d_data.shape)
print(l_data[img_num])
#print(h_data.shape)
#h_data[0] = np.flip(h_data[0])
h_data[img_num] = np.swapaxes(h_data[img_num], 0 ,1)
#h_data[img_num] = np.rot90(h_data[img_num])
#h_data[img_num] = np.flip(h_data[img_num], axis=0)
#h_data[0] = np.flip(h_data[0], axis=1)
#h_data[0] = np.flip(h_data[0])
img = h_data[img_num][:,:,0]
print("heat_data: ", len(h_data), h_data[img_num][l_data[img_num,0], l_data[img_num,1], 0])
depth = d_data[img_num]/255
normalized = normalize_fixed(depth)
#depth_uni = np.where(depth < 0.864, 1.0, 0.0).astype('float32')
kernel = (np.ones((7,7),np.float32)*10)/42
dst = cv2.filter2D(img ,-1,kernel)
print("GREATER 0: ", dst[dst > 0])
#dst = np.flip(dst)
rgb_data = rgb_data/255
f, axarr = plt.subplots(2,2)
axarr[0,0].imshow(img)
axarr[0,1].imshow(dst)
axarr[1,0].imshow(depth)
axarr[1,1].imshow(normalized)


#plt.subplot(211),plt.imshow(dst),plt.title('Averaging')
#plt.xticks([]), plt.yticks([])
plt.show()