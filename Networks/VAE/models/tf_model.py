import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
tfd = tfp.distributions
tfpl = tfp.layers
tfk = tf.keras
tfkl = tf.keras.layers
np.random.seed(25)

class create_Architecture():
    def __init__(self, indipendent, input_shape=(256,256,1), latent_dim=32):

        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.decoder_latent_dim = latent_dim
        '''ENCODER'''
        self.encoder_input = tfk.Input(shape=self.input_shape)
        self.encoder_noise = tfkl.GaussianNoise(0.1)
        self.encoder_conv_1 = tfkl.Conv2D(filters=32, kernel_size=3, strides=2, padding='same')
        self.encoder_conv_2 = tfkl.Conv2D(filters=64, kernel_size=3, strides=2, padding='same')
        self.encoder_conv_3 = tfkl.Conv2D(filters=64, kernel_size=3, strides=2, padding='same')
        self.encoder_conv_4 = tfkl.Conv2D(filters=128, kernel_size=3, strides=2, padding='same')
        self.encoder_batch_1 = tfkl.BatchNormalization()
        self.encoder_batch_2 = tfkl.BatchNormalization()
        self.encoder_batch_3 = tfkl.BatchNormalization()
        self.encoder_batch_4 = tfkl.BatchNormalization()
        self.encoder_pooling_1 = tfkl.MaxPooling2D(pool_size=2, padding="valid")
        self.encoder_pooling_2 = tfkl.MaxPooling2D(pool_size=2, padding="valid")
        self.encoder_lr_1 = tfkl.LeakyReLU(alpha=0.2)
        self.encoder_lr_2 = tfkl.LeakyReLU(alpha=0.2)
        self.encoder_lr_3 = tfkl.LeakyReLU(alpha=0.2)
        self.encoder_lr_4 = tfkl.LeakyReLU(alpha=0.2)
        self.encoder_multivariat_dense = tfkl.Dense(tfpl.MultivariateNormalTriL.params_size(self.latent_dim))
        self.encoder_multivariat_latent = tfpl.MultivariateNormalTriL(self.latent_dim, convert_to_tensor_fn=tfd.Distribution.sample, name='z_layer')
        self.encoder_prob_dense = tfkl.Dense(tfpl.IndependentNormal.params_size(self.latent_dim), activation=None, name='z_params')
        self.encoder_prob_latent = tfpl.IndependentNormal(self.latent_dim, convert_to_tensor_fn=tfd.Distribution.sample, name='z_layer')
        self.encoder_flatten_layer = tfkl.Flatten()
        '''DECODER'''
        self.latent_input = tfk.Input(shape=self.decoder_latent_dim)
        self.decoder_dense = tfkl.Dense(8 * 8 * 128, activation=None)
        self.decoder_reshape_layer = tfkl.Reshape((8, 8, 128))
        self.decoder_deconv_1 = tfkl.Conv2DTranspose(filters=128, kernel_size=3, strides=2, padding='same')
        self.decoder_deconv_2 = tfkl.Conv2DTranspose(filters=128, kernel_size=3, strides=2, padding='same')
        self.decoder_deconv_3 = tfkl.Conv2DTranspose(filters=64, kernel_size=3, strides=2, padding='same')
        self.decoder_deconv_4 = tfkl.Conv2DTranspose(filters=32, kernel_size=3, strides=2, padding='same')
        self.decoder_lr_1 = tfkl.LeakyReLU(alpha=0.2)
        self.decoder_lr_2 = tfkl.LeakyReLU(alpha=0.2)
        self.decoder_lr_3 = tfkl.LeakyReLU(alpha=0.2)
        self.decoder_lr_4 = tfkl.LeakyReLU(alpha=0.2)
        self.decoder_lr_5 = tfkl.LeakyReLU(alpha=0.2)
        self.decoder_batch_1 = tfkl.BatchNormalization()
        self.decoder_batch_2 = tfkl.BatchNormalization()
        self.decoder_batch_3 = tfkl.BatchNormalization()
        self.decoder_batch_4 = tfkl.BatchNormalization()
        self.decoder_flatten_layer = tfkl.Flatten(name='x_params')
        if indipendent == "normal":
            self.decoder_deconv_5 = tfkl.Conv2DTranspose(filters=2, kernel_size=3, strides=1, padding='same')
            self.decoder_prob_output = tfpl.IndependentNormal(self.input_shape, convert_to_tensor_fn=tfd.Distribution.sample, name='x_layer')
        else:
            self.decoder_deconv_5 = tfkl.Conv2DTranspose(filters=1, kernel_size=3, strides=1, padding='same')
            self.decoder_prob_output = tfpl.IndependentBernoulli(self.input_shape,convert_to_tensor_fn=tfd.Distribution.sample, name='x_layer')

    def leaky_relu(self, x):
        return tf.nn.leaky_relu(x, alpha=0.3)

    def create_prior(self):
        return tfd.Independent(tfd.Normal(loc=tf.zeros(self.latent_dim), scale=1.),reinterpreted_batch_ndims=1)

    def create_encoder(self):
        z = self.encoder_noise(self.encoder_input)
        z = self.encoder_conv_1(z)
        z = self.encoder_batch_1(z)
        z = self.encoder_lr_1(z)
        z = self.encoder_pooling_1(z)
        z = self.encoder_conv_2(z)
        z = self.encoder_batch_2(z)
        z = self.encoder_lr_2(z)
        z = self.encoder_conv_3(z)
        z = self.encoder_batch_3(z)
        z = self.encoder_lr_3(z)
        z = self.encoder_pooling_2(z)
        z = self.encoder_conv_4(z)
        z = self.encoder_batch_4(z)
        z = self.encoder_lr_4(z)
        z = self.encoder_flatten_layer(z)
        z = self.encoder_prob_dense(z)
        z = self.encoder_prob_latent(z)
        return tfk.models.Model(self.encoder_input, z, name="encoder")

    def create_decoder(self):
        x_output = self.decoder_dense(self.latent_input)
        x_output = self.decoder_reshape_layer(x_output)
        x_output = self.decoder_deconv_1(x_output)
        x_output = self.decoder_batch_1(x_output)
        x_output = self.decoder_lr_1(x_output)
        x_output = self.decoder_deconv_3(x_output)
        x_output = self.decoder_batch_3(x_output)
        x_output = self.decoder_lr_3(x_output)
        x_output = self.decoder_deconv_4(x_output)
        x_output = self.decoder_batch_4(x_output)
        x_output = self.decoder_lr_4(x_output)
        x_output = self.decoder_deconv_5(x_output)
        x_flatten = self.decoder_flatten_layer(x_output)
        #x_dense = self.decoder_normal_dense(x_flatten)
        prob_output = self.decoder_prob_output(x_flatten)
        return tfk.models.Model(self.latent_input, prob_output, name="decoder")


class VAE(tfk.Model):
    def __init__(self, encoder, decoder, kl_weight, latent_dim, is_3d=False, is_astar=False, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = tfk.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = tfk.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = tfk.metrics.Mean(name="kl_loss")
        self.val_total_loss_tracker = tfk.metrics.Mean(name="total_loss")
        self.val_reconstruction_loss_tracker = tfk.metrics.Mean(name="reconstruction_loss")
        self.val_kl_loss_tracker = tfk.metrics.Mean(name="kl_loss")
        self.kl_weight = kl_weight
        self.latent_dim = latent_dim
        self.is_3d = is_3d
        self.is_astar = is_astar

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
            self.val_total_loss_tracker,
            self.val_reconstruction_loss_tracker,
            self.val_kl_loss_tracker,
        ]

    def elbo_cost(self,z_prior, x_true, z, reconstruction, kl_weight=1):
        neg_log_likelihood = -reconstruction.log_prob(x_true)
        kl_d = tfd.kl_divergence(z, z_prior)
        elbo_local = -(kl_weight * kl_d + neg_log_likelihood)
        elbo = tf.reduce_mean(input_tensor=elbo_local)
        return -elbo, kl_d, neg_log_likelihood


    @tf.function
    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_prior = tfd.Independent(tfd.Normal(loc=tf.zeros(self.latent_dim), scale=1.), reinterpreted_batch_ndims=1)
            z = self.encoder(data)
            reconstruction = self.decoder(z)
            total_loss, kl_loss, reconstruction_loss = self.elbo_cost(z_prior, data, z, reconstruction, self.kl_weight)
        grads = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        #print(self.total_loss_tracker.result)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }


    @tf.function
    def test_step(self, data):
        z_prior = tfd.Independent(tfd.Normal(loc=tf.zeros(self.latent_dim), scale=1.), reinterpreted_batch_ndims=1)
        z = self.encoder(data)
        reconstruction = self.decoder(z)
        total_loss, kl_loss, reconstruction_loss = self.elbo_cost(z_prior, data, z, reconstruction, self.kl_weight)
        self.val_total_loss_tracker.update_state(total_loss)
        self.val_reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.val_kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.val_total_loss_tracker.result(),
            "reconstruction_loss": self.val_reconstruction_loss_tracker.result(),
            "kl_loss": self.val_kl_loss_tracker.result(),
        }
