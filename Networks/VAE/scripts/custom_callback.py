import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import matplotlib
import shutil
matplotlib.use('Agg')
import matplotlib.pyplot as plt
tfd = tfp.distributions
tfpl = tfp.layers
tfk = tf.keras
tfkl = tf.keras.layers
np.random.seed(25)


class SaveModelCallback(tfk.callbacks.Callback):
    def __init__(self, m_sp, p_sp, para_sp, ckpt_path, sample_data, img_size, m_sr=3, p_sr=3, sample_num=9):
        super(SaveModelCallback, self).__init__()
        self.parameters_save_path = para_sp
        self.model_save_rate = m_sr
        self.model_save_path = m_sp
        self.pic_save_rate = p_sr
        self.pic_save_path = p_sp
        self.ckpt_path = ckpt_path
        self.ckpt = None
        self.ckpt_manager = None
        self.sample_data = sample_data.reshape((sample_data.shape[0], sample_data.shape[1], sample_data.shape[2], 1))
        self.random_generator = np.random.RandomState()
        self.img_size = img_size
        self.sample_num = sample_num

    def normalize_fixed(self, x):
        x = x.astype(np.float)/255
        current_min, current_max = np.amin(x), np.amax(x)  # tf.expand_dims(current_range[:, 0], 1), tf.expand_dims(current_range[:, 1], 1)
        if current_min==current_max:
            return x*0+0.000001
        normed_min, normed_max = 0, 1  # tf.expand_dims(normed_range[:, 0], 1), tf.expand_dims(normed_range[:, 1], 1)
        x_normed = (x - current_min) / (current_max - current_min)
        x_normed = x_normed * (normed_max - normed_min) + normed_min
        return x_normed

    def on_epoch_begin(self, epoch, logs=None):
        self.model.encoder.layers[1].stddev = np.random.uniform(0, 1)
        print('updating sttdev in training')
        print(self.model.encoder.layers[1].stddev)

    def int_to_index(self, i, r , c):
        return int(i/r), int(i%c)


    def on_epoch_end(self, epoch, logs=None):
        self.ckpt_manager.save()
        if (epoch + 1) % self.model_save_rate == 0:
            print("save model to: ", self.model_save_path, " for epoch: ", epoch)
            tf.saved_model.save(self.model, self.model_save_path)
        if epoch % self.pic_save_rate == 0:
            print("save eval picture to: ", self.pic_save_path, " for epoch: ", epoch)
            sample_image_id = self.random_generator.randint(len(self.sample_data), size=self.sample_num)
            output_imgs = np.zeros((self.sample_num, self.img_size, self.img_size), dtype=np.float32)
            for i, id in enumerate(sample_image_id):
                test_data = self.sample_data[id]
                test_data_processed = self.normalize_fixed(test_data)#tf.image.convert_image_dtype(test_data, tf.float32)
                latent = self.model.encoder(tf.convert_to_tensor(test_data_processed[None, :], dtype=tf.float32))
                output = self.model.decoder(latent).sample().numpy()
                output_imgs[i] = output.reshape(self.img_size, self.img_size)
            fig, axs = plt.subplots(3, 3)
            #save generated imgs
            for i, img in enumerate(output_imgs):
                r, c = self.int_to_index(i, 3, 3)
                axs[r,c].imshow(output_imgs[i])
            plt.savefig(self.pic_save_path+"rec_sample_images_epoch_"+str(epoch+1))
            #save orig imgs
            for i, id in enumerate(sample_image_id):
                r, c =self.int_to_index(i, 3, 3)
                axs[r,c].imshow(self.sample_data[sample_image_id[i]])
            plt.savefig(self.pic_save_path+"sample_images_epoch_"+str(epoch+1))
            plt.close(fig)

    def on_train_begin(self, logs=None):
        self.ckpt = tf.train.Checkpoint(optimizer=self.model.optimizer, model=self.model)
        self.ckpt_manager = tf.train.CheckpointManager(self.ckpt, directory=self.ckpt_path, max_to_keep=1)
        self.ckpt.restore(self.ckpt_manager.latest_checkpoint)
        if self.ckpt_manager.latest_checkpoint:
            print("Restored From {}".format(self.ckpt_manager.latest_checkpoint))
        else:
            print("Initializing from scratch")
        shutil.copyfile("parameters.yaml", self.model_save_path + "param_file.py")


class SaveModelCallback3D(tfk.callbacks.Callback):
    def __init__(self, m_sp, p_sp, para_sp, ckpt_path, sample_depth, sample_heat, sample_coord, img_size, m_sr=3, p_sr=3, sample_num=9):
        super(SaveModelCallback3D, self).__init__()
        self.parameters_save_path = para_sp
        self.model_save_rate = m_sr
        self.model_save_path = m_sp
        self.pic_save_rate = p_sr
        self.pic_save_path = p_sp
        self.ckpt_path = ckpt_path
        self.ckpt = None
        self.ckpt_manager = None
        self.sample_depth = sample_depth.reshape((sample_depth.shape[0], sample_depth.shape[1], sample_depth.shape[2], 1))
        self.sample_heat = sample_heat
        self.sample_coord = sample_coord
        self.random_generator = np.random.RandomState()
        self.img_size = img_size
        self.sample_num = sample_num

    def normalize_fixed(self, x):
        x = x.astype(np.float)/255
        current_min, current_max = np.amin(x), np.amax(x)  # tf.expand_dims(current_range[:, 0], 1), tf.expand_dims(current_range[:, 1], 1)
        normed_min, normed_max = 0, 1  # tf.expand_dims(normed_range[:, 0], 1), tf.expand_dims(normed_range[:, 1], 1)
        x_normed = (x - current_min) / (current_max - current_min)
        x_normed = x_normed * (normed_max - normed_min) + normed_min
        return x_normed

    def on_epoch_begin(self, epoch, logs=None):
        self.model.encoder.layers[1].stddev = np.random.uniform(0, 1)
        print('updating sttdev in training')
        print(self.model.encoder.layers[1].stddev)

    def int_to_index(self, i, r , c):
        return int(i/r), int(i%c)


    def on_epoch_end(self, epoch, logs=None):
        self.ckpt_manager.save()
        if (epoch + 1) % self.model_save_rate == 0:
            print("save model to: ", self.model_save_path, " for epoch: ", epoch)
            tf.saved_model.save(self.model, self.model_save_path)
        if epoch % self.pic_save_rate == 0:
            print("save eval picture to: ", self.pic_save_path, " for epoch: ", epoch)
            sample_image_id = self.random_generator.randint(len(self.sample_depth), size=self.sample_num)
            output_imgs = np.zeros((self.sample_num, self.img_size, self.img_size), dtype=np.float32)
            for i, id in enumerate(sample_image_id):
                test_data = self.sample_depth[id]
                test_coord = tf.convert_to_tensor(self.sample_coord[id].reshape(1, 4) / 255, dtype=tf.float32)
                test_data_processed = self.normalize_fixed(test_data)#tf.image.convert_image_dtype(test_data, tf.float32)
                latent = self.model.encoder(tf.convert_to_tensor(test_data_processed[None, :], dtype=tf.float32))
                extended_latent = tf.concat((latent, test_coord), axis=1)
                output = self.model.decoder(extended_latent).mean().numpy()
                output_imgs[i] = output
            fig, axs = plt.subplots(3, 3)
            #save generated imgs
            self.plt_loop(output_imgs, axs, "heat", epoch)
            self.plt_loop(output_imgs, axs, "yaw", epoch)
            self.plt_loop(output_imgs, axs, "base", epoch)
            fig.close()

    def plt_loop(self, data, axs, what, epoch):
        for i, img in enumerate(data):
            r, c = self.int_to_index(i, 3, 3)
            if what == "heat":
                axs[r, c].imshow(data[i][:, :, 0])
                plt.savefig(self.pic_save_path+"rec_sample_images_heat_epoch_"+str(epoch+1))
            elif what == "yaw":
                axs[r, c].imshow(data[i][:, :, 1])
                plt.savefig(self.pic_save_path + "rec_sample_images_yaw_epoch_" + str(epoch + 1))
            else:
                axs[r, c].imshow(self.sample_data[data[i]])
                plt.savefig(self.pic_save_path + "sample_images_epoch_" + str(epoch + 1))
        return



    def on_train_begin(self, logs=None):
        self.ckpt = tf.train.Checkpoint(optimizer=self.model.optimizer, model=self.model)
        self.ckpt_manager = tf.train.CheckpointManager(self.ckpt, directory=self.ckpt_path, max_to_keep=1)
        self.ckpt.restore(self.ckpt_manager.latest_checkpoint)
        if self.ckpt_manager.latest_checkpoint:
            print("Restored From {}".format(self.ckpt_manager.latest_checkpoint))
        else:
            print("Initializing from scratch")
        shutil.copyfile("parameters.yaml", self.model_save_path + "param_file.py")


class SaveModelCallbackAstar(tfk.callbacks.Callback):
    def __init__(self, m_sp, p_sp, para_sp, ckpt_path, sample_depth, sample_astar, sample_coord, img_size, m_sr=3, p_sr=3, sample_num=9):
        super(SaveModelCallbackAstar, self).__init__()
        self.parameters_save_path = para_sp
        self.model_save_rate = m_sr
        self.model_save_path = m_sp
        self.pic_save_rate = p_sr
        self.pic_save_path = p_sp
        self.ckpt_path = ckpt_path
        self.ckpt = None
        self.ckpt_manager = None
        self.sample_depth = sample_depth.reshape((sample_depth.shape[0], sample_depth.shape[1], sample_depth.shape[2], 1))
        self.sample_astar = sample_astar
        self.sample_coord = sample_coord
        self.random_generator = np.random.RandomState()
        self.img_size = img_size
        self.sample_num = sample_num

    def normalize_fixed(self, x):
        x = x.astype(np.float)/255
        current_min, current_max = np.amin(x), np.amax(x)  # tf.expand_dims(current_range[:, 0], 1), tf.expand_dims(current_range[:, 1], 1)
        normed_min, normed_max = 0, 1  # tf.expand_dims(normed_range[:, 0], 1), tf.expand_dims(normed_range[:, 1], 1)
        x_normed = (x - current_min) / (current_max - current_min)
        x_normed = x_normed * (normed_max - normed_min) + normed_min
        return x_normed

    def on_epoch_begin(self, epoch, logs=None):
        self.model.encoder.layers[1].stddev = np.random.uniform(0, 1)
        print('updating sttdev in training')
        print(self.model.encoder.layers[1].stddev)

    def int_to_index(self, i, r , c):
        return int(i/r), int(i%c)


    def on_epoch_end(self, epoch, logs=None):
        self.ckpt_manager.save()
        if (epoch + 1) % self.model_save_rate == 0:
            print("save model to: ", self.model_save_path, " for epoch: ", epoch)
            tf.saved_model.save(self.model, self.model_save_path)
        if epoch % self.pic_save_rate == 0:
            print("save eval picture to: ", self.pic_save_path, " for epoch: ", epoch)
            sample_image_id = self.random_generator.randint(len(self.sample_depth), size=self.sample_num)
            output_imgs = np.zeros((self.sample_num, self.img_size, self.img_size), dtype=np.float32)
            for i, id in enumerate(sample_image_id):
                test_data = self.sample_depth[id]
                test_coord = tf.convert_to_tensor(self.sample_coord[id].reshape(1, 4) / 255, dtype=tf.float32)
                test_data_processed = self.normalize_fixed(test_data)#tf.image.convert_image_dtype(test_data, tf.float32)
                latent = self.model.encoder(tf.convert_to_tensor(test_data_processed[None, :], dtype=tf.float32))
                extended_latent = tf.concat((latent, test_coord), axis=1)
                output = self.model.decoder(extended_latent).mean().numpy()
                output_imgs[i] = output.reshape(256,256)
            fig, axs = plt.subplots(3, 3)
            #save generated imgs
            self.plt_loop(output_imgs, axs, "heat", epoch)
            self.plt_loop(sample_image_id, axs, "orig_heat", epoch)
            self.plt_loop(sample_image_id, axs, "orig", epoch)
            plt.close(fig)

    def plt_loop(self, data, axs, what, epoch):
        for i, img in enumerate(data):
            r, c = self.int_to_index(i, 3, 3)
            if what == "heat":
                axs[r, c].imshow(data[i])
                plt.savefig(self.pic_save_path+"rec_sample_images_heat_epoch_"+str(epoch+1))
            elif what == "orig":
                axs[r, c].imshow(self.sample_depth[data[i]])
                plt.savefig(self.pic_save_path + "sample_images_epoch_" + str(epoch + 1))
            elif what == "orig_heat":
                axs[r, c].imshow(self.sample_astar[data[i]])
                plt.savefig(self.pic_save_path + "sample_images_orig_heat_epoch_" + str(epoch + 1))
        return



    def on_train_begin(self, logs=None):
        self.ckpt = tf.train.Checkpoint(optimizer=self.model.optimizer, model=self.model)
        self.ckpt_manager = tf.train.CheckpointManager(self.ckpt, directory=self.ckpt_path, max_to_keep=1)
        self.ckpt.restore(self.ckpt_manager.latest_checkpoint)
        if self.ckpt_manager.latest_checkpoint:
            print("Restored From {}".format(self.ckpt_manager.latest_checkpoint))
        else:
            print("Initializing from scratch")
        shutil.copyfile("parameters.yaml", self.model_save_path + "param_file.py")

