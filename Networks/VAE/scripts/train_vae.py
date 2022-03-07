from tf_model import VAE, create_Architecture
from custom_callback import SaveModelCallback
import tensorflow as tf
import numpy as np
import h5py
import os
import cv2
tfk = tf.keras
from tqdm import tqdm
#os.environ["CUDA_VISIBLE_DEVICES"] = str(5)
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)


def preprocess_images(image):
    unify = tf.image.convert_image_dtype(image, tf.float32)
    current_min, current_max = tf.experimental.numpy.amin(unify), tf.experimental.numpy.amax(unify)
    if current_min == current_max:
        return unify*0
    normed_min, normed_max = 0, 1
    x_normed = (unify - current_min) / (current_max - current_min)
    x_normed = x_normed * (normed_max - normed_min) + normed_min
    noise = tf.random.normal(shape=tf.shape(unify), mean=0.0, stddev=np.random.uniform(0,1), dtype=tf.float32)
    return tf.add(x_normed, noise)


def create_dataset(input_data, batch_size):
    train_data = input_data.reshape((input_data.shape[0], input_data.shape[1], input_data.shape[2], 1))
    train_ds = tf.data.Dataset.from_tensor_slices(train_data)
    train_ds = train_ds.map(lambda x: tf.py_function(func=preprocess_images, inp=[x], Tout=tf.float32),
                            num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)
    return train_ds.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)


def create_all_paths(paths):
    for p in paths:
        os.makedirs(p, exist_ok=True)

def train_model(data_file_path, model_save_path, checkpoint_save_path, eval_picture_save_path, parameter_save_path,
                batch_size, learning_rate, latent_dim, kl_weight, model_save_rate,
                picture_save_rate, independent, epochs=100, img_size=256):
    print(learning_rate)
    create_all_paths([model_save_path, checkpoint_save_path, eval_picture_save_path])
    # create train and test dataset
    print("START LOADING H5")
    sample_img_len = 500
    h5f_data = h5py.File(os.path.join(os.path.dirname(__file__), data_file_path), "r")
    print("CONVERTING TO NP")
    np_data = np.zeros((len(h5f_data), img_size, img_size), dtype=np.uint8)
    for i in tqdm(range(len(h5f_data))):
        dset = h5f_data["data_"+str(i)]
        np_data[i] = np.asarray(dset[:])
    print("DONE WITH CONVERTING")
    # split data in train and qualitative test set
    train_data, test_data = (
        np_data[:int(len(np_data)), :],
        np_data[int(len(np_data) - sample_img_len):, :],)
    print("START CREATING TF DATASET")
    train_dataset = create_dataset(train_data, batch_size)
    train_data = "Free"
    print("DONE CREATING DATASET")
    # create encoder and decoder
    architecture = create_Architecture(independent, input_shape=(img_size, img_size, 1), latent_dim=latent_dim)
    encoder = architecture.create_encoder()
    decoder = architecture.create_decoder()
    #encoder.summary()
    #decoder.summary()
    # create Model
    vae = VAE(encoder, decoder, kl_weight, latent_dim)
    #vae.encoder.trainable = False  #
    opt = tfk.optimizers.Adam(learning_rate=float(learning_rate))
    vae.compile(optimizer=opt)
    vae.fit(train_dataset, epochs=epochs, batch_size=batch_size, callbacks=[SaveModelCallback(model_save_path,
                                                                                              eval_picture_save_path,
                                                                                              parameter_save_path,
                                                                                              checkpoint_save_path,
                                                                                              test_data,
                                                                                              img_size,
                                                                                              m_sr=model_save_rate,
                                                                                              p_sr=picture_save_rate)])

