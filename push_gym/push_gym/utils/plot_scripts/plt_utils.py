import numpy as np
import matplotlib.pyplot as plt
import cv2

def get_bin_index_from_int(index, col_dim):
    row = index / col_dim
    col = index % col_dim
    return int(row), int(col)

def plt_img(imgs, row_dim, col_dim, save=False, path=None):
    fig, axs = plt.subplots(row_dim, col_dim)
    for idx, im in enumerate(imgs):
        r, c = get_bin_index_from_int(idx, col_dim)
        axs[r, c].imshow(im)
    if save:  plt.savefig(pic_save_path + "rec_sample_images_" + layer + "_epoch_" + str(epoch + 1))

def plt_multi_layer(imgs, row_dim, col_dim):
    fig, axs = plt.subplots(row_dim, col_dim)
    fig2, axs2 = plt.subplots(row_dim, col_dim)
    for idx, im in enumerate(imgs):
        r, c = get_bin_index_from_int(idx, col_dim)
        axs[r, c].imshow(im[:, :, 0])
        axs2[r, c].imshow(im[:, :, 1])


def plt_single_img_multi_layer(img, save=False, path=None, name=None):
    fig, axs = plt.subplots(2)
    axs[0].imshow(img[:, :, 0])
    axs[1].imshow(img[:, :, 1])
    if save:
        plt.savefig( path+name)
        #print("saved to", path+name+".png")

def plt_single_img(img, save=False, path=None, name=None):
    plt.imshow(img)
    if save:
        plt.savefig( path+name)
        #print("saved to", path+name+".png")

def save_plot_img(plt_fig, img, current_iteration_step):
    if plt_fig is None:
        plt_obj = []
        plt_fig = plt.figure()
        plt_fig.imshow(img)
        plt.savefig("~/test_saving/save_" + str(current_iteration_step) + ".png")
    else:
        plt_fig.set_data(img)
        plt_fig.canvas.draw()
        plt.savefig("~/test_saving/save_" + str(current_iteration_step) + ".png")
    return plt_fig

def plotImage(image_size, plt_num, plt_fig, plt_obj, path, depth, lw,  recon, arrow=None, text=None, reward=0):
    if plt_fig is None:

        plt_obj = []
        plt_fig = plt.figure()#figsize=(1, 2))

        # Plot truth
        if path is not None:
            reward = round(reward, 5)
            path = cv2.putText(path, str(reward), (150, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0),3, cv2.LINE_AA)
            plt.subplot(3, 2, 1)
            plt_obj.append(plt.imshow(path))
            #plt.axis("off")
            plt.show(block=False)

        if depth is not None:
            plt.subplot(3,2, 2)
            plt_obj.append(plt.imshow(depth))
            #plt.axis("off")
            plt.show(block=False)

        # Plot reconstruction
        if recon is not None:
            plt.subplot(3, 2, 3)
            plt_obj.append(plt.imshow(recon))
        else:
            plt_obj.append(plt.imshow(np.ones((image_size,image_size))))
        #plt.axis("off")
        plt.show(block=False)

        if lw is not None:
            plt.subplot(3, 2, 4)
            plt_obj.append(plt.imshow(lw))
            plt.show(block=False)
        if arrow is not None:
            plt.subplot(3, 2, 5)
            plt_obj.append(plt.imshow(arrow))
            plt.show(block=False)
        if text is not None:
            plt.subplot(3, 2, 6)
            plt_obj.append(plt.imshow(text))
            plt.show(block=False)
    else:
        mng = plt.get_current_fig_manager()
        mng.resize(800,800)
        if path is not None:
            reward = round(reward, 5)
            path = cv2.putText(path, str(reward), (150, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0),3, cv2.LINE_AA)
            plt_obj[0].set_data(path)
        if depth is not None:
            plt_obj[1].set_data(depth)
        if recon is not None:
            plt_obj[2].set_data(recon)
        if lw is not None:
            plt_obj[3].set_data(lw)
        if arrow is not None:
            plt_obj[4].set_data(arrow)
        if text is not None:
            plt_obj[5].set_data(text)
        plt_fig.canvas.draw()
    return plt_fig, plt_obj
